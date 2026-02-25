# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional  # 导入列表和可选类型提示
import torch  # 导入 PyTorch 深度学习框架
import os  # 导入操作系统接口模块

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper  # 从工具模块导入 Wan 模型的各个组件封装

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation, log_gpu_memory  # 导入显存管理相关的工具函数
from utils.debug_option import DEBUG  # 导入调试选项配置
import torch.distributed as dist  # 导入 PyTorch 分布式训练模块

class CausalInferencePipeline(torch.nn.Module):  # 定义因果推理管道类，继承自 torch.nn.Module
    def __init__(  # 初始化函数
            self,  # 类实例本身
            args,  # 参数对象
            device,  # 运行设备（如 GPU）
            generator=None,  # 生成器模型实例
            text_encoder=None,  # 文本编码器实例

            vae=None,  # VAE 模型实例
            image_encoder=None,  # 图像编码器实例
    ):
        super().__init__()  # 调用父类的初始化方法
        # Step 1: Initialize all models  # 第一步：初始化所有模型组件
        if DEBUG:  # 如果开启了调试模式
            print(f"args.model_kwargs: {args.model_kwargs}")  # 打印模型参数配置
        self.generator = WanDiffusionWrapper(  # 初始化或赋值生成器模型
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator  # 如果未传入 generator 则创建新的，启用因果模式
        
        model_kwargs = getattr(args, "model_kwargs", {})  # 获取模型参数字典，默认为空
        model_name = model_kwargs.get("model_name", "Wan2.1-T2V-1.3B")  # 获取模型名称，默认为 Wan2.1-T2V-1.3B

        self.text_encoder = WanTextEncoder(model_name=model_name) if text_encoder is None else text_encoder  # 初始化或赋值文本编码器
        self.vae = WanVAEWrapper(model_name=model_name) if vae is None else vae  # 初始化或赋值 VAE 模型
        # initialize image encoder for i2v  # 为 I2V 任务初始化图像编码器
        self.image_encoder = None  # 初始化图像编码器为 None
        if getattr(args, "i2v", False):  # 如果参数中指定了 I2V 模式
            from utils.wan_wrapper import WanCLIPEncoder  # 延迟导入 WanCLIPEncoder
            self.image_encoder = WanCLIPEncoder(model_name=model_name) if image_encoder is None else image_encoder  # 初始化或赋值图像编码器

        # Step 2: Initialize all causal hyperparmeters  # 第二步：初始化所有因果推理相关的超参数
        self.scheduler = self.generator.get_scheduler()  # 获取生成器的调度器
        self.denoising_step_list = torch.tensor(  # 将去噪步数列表转换为 Tensor
            args.denoising_step_list, dtype=torch.long)  # 数据类型为长整型
        if args.warp_denoising_step:  # 如果开启了 warp 去噪步数
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))  # 拼接时间步和 0
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]  # 根据索引重新映射去噪步数

        # hard code for Wan2.1-T2V-1.3B  # 针对 Wan2.1-T2V-1.3B 模型的硬编码参数
        self.num_transformer_blocks = 30  # Transformer 块的数量
        self.frame_seq_length = 1560  # 每帧的序列长度

        self.kv_cache1 = None  # 初始化 KV 缓存为 None
        self.args = args  # 保存参数对象
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)  # 获取每个块的帧数，默认为 1
        self.local_attn_size = args.model_kwargs.local_attn_size  # 获取局部注意力大小

        # Normalize to list if sequence-like (e.g., OmegaConf ListConfig)  # 规范化为列表（如果它是序列类型）

        if not dist.is_initialized() or dist.get_rank() == 0:  # 如果未开启分布式或当前是主进程
            print(f"KV inference with {self.num_frame_per_block} frames per block")  # 打印推理块帧数信息

        if self.num_frame_per_block > 1:  # 如果每个块的帧数大于 1
            self.generator.model.num_frame_per_block = self.num_frame_per_block  # 设置生成器模型的块帧数

    def inference(  # 定义推理方法
        self,  # 类实例
        noise: torch.Tensor,  # 输入噪声张量
        text_prompts: List[str],  # 文本提示列表
        return_latents: bool = False,  # 是否返回潜在变量
        profile: bool = False,  # 是否开启性能分析
        low_memory: bool = False,  # 是否开启低显存模式
        clip_fea: Optional[torch.Tensor] = None,  # CLIP 特征（可选）
        y: Optional[torch.Tensor] = None,  # 条件 y（可选）
        kv_cache: Optional[List] = None,  # KV 缓存（可选）
        crossattn_cache: Optional[List] = None,  # 交叉注意力缓存（可选）
        start_frame_idx: int = 0,  # 起始帧索引
    ) -> torch.Tensor:  # 返回张量
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape  # 解包噪声张量的形状维度
        print(f"num_output_frames: {num_output_frames}, num_frame_per_block: {self.num_frame_per_block}")  # 打印输出帧数和块大小
        assert num_output_frames % self.num_frame_per_block == 0  # 断言输出帧数能被块大小整除
        num_blocks = num_output_frames // self.num_frame_per_block  # 计算总块数

        conditional_dict = self.text_encoder(  # 调用文本编码器获取条件字典
            text_prompts=text_prompts  # 传入文本提示
        )

        if low_memory:  # 如果开启低显存模式
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5  # 计算保留显存大小
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)  # 移动文本编码器到 GPU 并保留显存

        # Decide the device for output based on low_memory (CPU for low-memory mode; otherwise GPU)  # 根据低显存模式决定输出设备
        output_device = torch.device('cpu') if low_memory else noise.device  # 确定输出设备
        output = torch.zeros(  # 初始化输出张量为全零
            [batch_size, num_output_frames, num_channels, height, width],  # 定义输出形状
            device=output_device,  # 指定输出设备
            dtype=noise.dtype  # 指定数据类型与噪声一致
        )

        # Set up profiling if requested  # 如果请求性能分析，则设置相关事件
        if profile:  # 如果开启性能分析
            init_start = torch.cuda.Event(enable_timing=True)  # 初始化开始事件
            init_end = torch.cuda.Event(enable_timing=True)  # 初始化结束事件
            diffusion_start = torch.cuda.Event(enable_timing=True)  # 扩散过程开始事件
            diffusion_end = torch.cuda.Event(enable_timing=True)  # 扩散过程结束事件
            vae_start = torch.cuda.Event(enable_timing=True)  # VAE 开始事件
            vae_end = torch.cuda.Event(enable_timing=True)  # VAE 结束事件
            block_times = []  # 块时间列表
            block_start = torch.cuda.Event(enable_timing=True)  # 块开始事件
            block_end = torch.cuda.Event(enable_timing=True)  # 块结束事件
            init_start.record()  # 记录初始化开始时间

        # Step 1: Initialize KV cache to all zeros  # 第一步：初始化 KV 缓存为全零
        if kv_cache is not None:  # 如果传入了 KV 缓存
             print(f"[inference] Reusing provided KV cache (streaming mode)")  # 打印重用缓存信息
             self.kv_cache1 = kv_cache  # 使用传入的 KV 缓存
        else:  # 否则
            local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)  # 获取局部注意力配置
            kv_policy = ""  # 初始化 KV 策略字符串
            if local_attn_cfg != -1:  # 如果是局部注意力
                # local attention  # 局部注意力模式
                kv_cache_size = local_attn_cfg * self.frame_seq_length  # 计算 KV 缓存大小
                kv_policy = f"int->local, size={local_attn_cfg}"  # 设置策略描述
            else:  # 否则
                # global attention  # 全局注意力模式
                kv_cache_size = num_output_frames * self.frame_seq_length  # 计算全局缓存大小
                kv_policy = "global (-1)"  # 设置策略描述
            print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")  # 打印缓存配置信息

            self._initialize_kv_cache(  # 初始化 KV 缓存
                batch_size=batch_size,  # 批次大小
                dtype=noise.dtype,  # 数据类型
                device=noise.device,  # 设备
                kv_cache_size_override=kv_cache_size  # 覆盖缓存大小
            )
            
        if crossattn_cache is not None:  # 如果传入了交叉注意力缓存
            self.crossattn_cache = crossattn_cache  # 使用传入的缓存
        else:  # 否则
            self._initialize_crossattn_cache(  # 初始化交叉注意力缓存
                batch_size=batch_size,  # 批次大小
                dtype=noise.dtype,  # 数据类型
                device=noise.device  # 设备
            )

        current_start_frame = start_frame_idx  # 设置当前起始帧索引
        self.generator.model.local_attn_size = self.local_attn_size  # 设置生成器模型的局部注意力大小
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")  # 打印局部注意力设置
        self._set_all_modules_max_attention_size(self.local_attn_size)  # 设置所有子模块的最大注意力大小

        if profile:  # 如果开启性能分析
            init_end.record()  # 记录初始化结束
            torch.cuda.synchronize()  # 同步 CUDA
            diffusion_start.record()  # 记录扩散开始

        # Step 2: Temporal denoising loop  # 第二步：时间去噪循环
        all_num_frames = [self.num_frame_per_block] * num_blocks  # 计算每个块的帧数列表
        for block_index, current_num_frames in enumerate(all_num_frames):  # 遍历每个块
            if profile:  # 如果开启性能分析
                block_start.record()  # 记录块开始

            noisy_input = noise[  # 获取当前块的噪声输入
                :, current_start_frame:current_start_frame + current_num_frames]  # 根据索引切片

            # Step 2.1: Spatial denoising loop  # 2.1 步：空间去噪循环
            for index, current_timestep in enumerate(self.denoising_step_list):  # 遍历去噪步数
                # print(f"current_timestep: {current_timestep}")  # 打印当前时间步

                # set current timestep  # 设置当前时间步
                timestep = torch.ones(  # 创建时间步张量
                    [batch_size, current_num_frames],  # 形状
                    device=noise.device,  # 设备
                    dtype=torch.int64) * current_timestep  # 数据类型和值

                if index < len(self.denoising_step_list) - 1:  # 如果不是最后一步
                    _, denoised_pred = self.generator(  # 调用生成器进行预测
                        noisy_image_or_video=noisy_input,  # 输入噪声
                        conditional_dict=conditional_dict,  # 条件字典
                        timestep=timestep,  # 时间步
                        kv_cache=self.kv_cache1,  # KV 缓存
                        crossattn_cache=self.crossattn_cache,  # 交叉注意力缓存
                        current_start=current_start_frame * self.frame_seq_length,  # 当前起始位置
                        clip_fea=clip_fea,  # CLIP 特征
                        # y slice: relative to the current window usually.   # y 切片：通常相对于当前窗口
                        # If y is passed as full length (unlikely), we need offset.  # 如果 y 是全长的（不常见），我们需要偏移
                        # In inference_i2v.py, we pass y corresponding to *current window's* ref.  # 在 inference_i2v.py 中，我们传入与当前窗口参考对应的 y
                        # So y is small [B, 4, 16, 60, 104].  # 所以 y 比较小
                        # Block loop inside here goes 0, 4, 8... relative to start of THIS window.  # 这里的块循环相对于窗口起始位置
                        # BUT current_start_frame is increasing globally.  # 但 current_start_frame 是全局增加的
                        # So if window 2, current_start_frame = 20.  # 所以如果是窗口 2，current_start_frame = 20
                        # y slice: u[:, (current_start_frame % num_output_frames_in_this_call) ...]  # y 切片逻辑
                        # This is getting tricky.   # 这变得有点复杂
                        # Let's simple fix: The passed `y` corresponds to the current call's noise.   # 简单修复：传入的 y 对应当前调用的噪声
                        # So we should use relative indexing for y.  # 所以我们应该对 y 使用相对索引
                        y=[u[:, (current_start_frame - start_frame_idx):(current_start_frame - start_frame_idx) + current_num_frames] for u in y] if y is not None else None  # 切片 y
                    )
                    next_timestep = self.denoising_step_list[index + 1]  # 获取下一个时间步
                    noisy_input = self.scheduler.add_noise(  # 添加噪声
                        denoised_pred.flatten(0, 1),  # 展平预测结果
                        torch.randn_like(denoised_pred.flatten(0, 1)),  # 生成随机噪声
                        next_timestep * torch.ones(  # 生成时间步张量
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)  # 形状和设备
                    ).unflatten(0, denoised_pred.shape[:2])  # 恢复形状
                else:  # 如果是最后一步
                    # for getting real output  # 获取真实输出
                    _, denoised_pred = self.generator(  # 调用生成器
                        noisy_image_or_video=noisy_input,  # 输入噪声
                        conditional_dict=conditional_dict,  # 条件字典
                        timestep=timestep,  # 时间步
                        kv_cache=self.kv_cache1,  # KV 缓存
                        crossattn_cache=self.crossattn_cache,  # 交叉注意力缓存
                        current_start=current_start_frame * self.frame_seq_length,  # 当前起始位置
                        clip_fea=clip_fea,  # CLIP 特征
                        y=[u[:, (current_start_frame - start_frame_idx):(current_start_frame - start_frame_idx) + current_num_frames] for u in y] if y is not None else None  # 切片 y
                    )
            # Step 2.2: record the model's output  # 2.2 步：记录模型输出
            # output buffer is local to this call, so 0-based indexing relative to this call's result  # 输出缓冲区是本地的，所以使用 0 基索引
            output[:, (current_start_frame - start_frame_idx):(current_start_frame - start_frame_idx) + current_num_frames] = denoised_pred.to(output.device)  # 将预测结果存入输出
            # Step 2.3: rerun with timestep zero to update KV cache using clean context  # 2.3 步：使用时间步 0 重新运行以更新 KV 缓存
            context_timestep = torch.ones_like(timestep) * self.args.context_noise  # 创建上下文时间步
            self.generator(  # 调用生成器
                noisy_image_or_video=denoised_pred,  # 输入去噪后的图像
                conditional_dict=conditional_dict,  # 条件字典
                timestep=context_timestep,  # 时间步
                kv_cache=self.kv_cache1,  # KV 缓存
                crossattn_cache=self.crossattn_cache,  # 交叉注意力缓存
                current_start=current_start_frame * self.frame_seq_length,  # 当前起始位置
                clip_fea=clip_fea,  # CLIP 特征
                y=[u[:, (current_start_frame - start_frame_idx):(current_start_frame - start_frame_idx) + current_num_frames] for u in y] if y is not None else None  # 切片 y
            )

            if profile:  # 如果开启性能分析
                block_end.record()  # 记录块结束
                torch.cuda.synchronize()  # 同步 CUDA
                block_time = block_start.elapsed_time(block_end)  # 计算块时间
                block_times.append(block_time)  # 添加到列表

            # Step 3.4: update the start and end frame indices  # 3.4 步：更新起始帧索引
            current_start_frame += current_num_frames  # 增加当前起始帧

        if profile:  # 如果开启性能分析
            # End diffusion timing and synchronize CUDA  # 结束扩散计时并同步
            diffusion_end.record()  # 记录扩散结束
            torch.cuda.synchronize()  # 同步 CUDA
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)  # 计算扩散时间
            init_time = init_start.elapsed_time(init_end)  # 计算初始化时间
            vae_start.record()  # 记录 VAE 开始

        # Step 3: Decode the output  # 第三步：解码输出
        if getattr(self.args.model_kwargs, "use_infinite_attention", False):  # 如果使用无限注意力
            video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)  # 分块解码
        else:  # 否则
            video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)  # 直接解码
        video = (video * 0.5 + 0.5).clamp(0, 1)  # 反归一化并截断
        if profile:  # 如果开启性能分析
            # End VAE timing and synchronize CUDA  # 结束 VAE 计时并同步
            vae_end.record()  # 记录 VAE 结束
            torch.cuda.synchronize()  # 同步 CUDA
            vae_time = vae_start.elapsed_time(vae_end)  # 计算 VAE 时间
            total_time = init_time + diffusion_time + vae_time  # 计算总时间

            print("Profiling results:")  # 打印性能分析结果
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")  # 打印初始化时间
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")  # 打印扩散时间
            for i, block_time in enumerate(block_times):  # 遍历块时间
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")  # 打印每块时间
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")  # 打印 VAE 时间
            print(f"  - Total time: {total_time:.2f} ms")  # 打印总时间

        if return_latents:  # 如果需要返回潜在变量
            return video, output.to(noise.device)  # 返回视频和潜在变量
        else:  # 否则
            return video  # 仅返回视频

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):  # 初始化 KV 缓存的方法
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []  # 初始化缓存列表
        # Determine cache size  # 确定缓存大小
        if kv_cache_size_override is not None:  # 如果有覆盖值
            kv_cache_size = kv_cache_size_override  # 使用覆盖值
        else:  # 否则
            if self.local_attn_size != -1:  # 如果是局部注意力
                # Local attention: cache only needs to store the window  # 局部注意力：缓存仅需存储窗口
                kv_cache_size = self.local_attn_size * self.frame_seq_length  # 计算大小
            else:  # 否则
                # Global attention: default cache for 21 frames (backward compatibility)  # 全局注意力：默认为 21 帧
                kv_cache_size = 32760  # 设置大小

        for _ in range(self.num_transformer_blocks):  # 遍历 Transformer 块
            kv_cache1.append({  # 添加缓存字典
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),  # 初始化 k
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),  # 初始化 v
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),  # 初始化全局结束索引
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)  # 初始化局部结束索引
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache  # 保存干净的缓存

    def _initialize_crossattn_cache(self, batch_size, dtype, device):  # 初始化交叉注意力缓存的方法
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []  # 初始化缓存列表

        for _ in range(self.num_transformer_blocks):  # 遍历 Transformer 块
            crossattn_cache.append({  # 添加缓存字典
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),  # 初始化 k
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),  # 初始化 v
                "is_init": False  # 初始化标志
            })
        self.crossattn_cache = crossattn_cache  # 保存缓存

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):  # 设置所有模块最大注意力大小的方法
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model's global default (32760 for Wan, 28160 for 5B).
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if local_attn_size_value == -1:  # 如果值为 -1
            target_size = 32760  # 设置为默认全局大小
            policy = "global"  # 策略为全局
        else:  # 否则
            target_size = int(local_attn_size_value) * self.frame_seq_length  # 计算目标大小
            policy = "local"  # 策略为局部

        updated_modules = []  # 初始化更新模块列表
        # Update root model if applicable  # 如果适用，更新根模型
        if hasattr(self.generator.model, "max_attention_size"):  # 如果根模型有该属性
            try:  # 尝试
                prev = getattr(self.generator.model, "max_attention_size")  # 获取旧值
            except Exception:  # 捕获异常
                prev = None  # 置空
            setattr(self.generator.model, "max_attention_size", target_size)  # 设置新值
            updated_modules.append("<root_model>")  # 添加到列表

        # Update all child modules  # 更新所有子模块
        for name, module in self.generator.model.named_modules():  # 遍历所有子模块
            if hasattr(module, "max_attention_size"):  # 如果模块有该属性
                try:  # 尝试
                    prev = getattr(module, "max_attention_size")  # 获取旧值
                except Exception:  # 捕获异常
                    prev = None  # 置空
                try:  # 尝试
                    setattr(module, "max_attention_size", target_size)  # 设置新值
                    updated_modules.append(name if name else module.__class__.__name__)  # 添加到列表
                except Exception:  # 捕获异常
                    pass  # 忽略