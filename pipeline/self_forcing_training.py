# Adopted from https://github.com/guandeh17/Self-Forcing  # 采用自 Self-Forcing 开源项目
# SPDX-License-Identifier: Apache-2.0  # 采用 Apache 2.0 开源协议
from utils.wan_wrapper import WanDiffusionWrapper  # 导入 Wan 模型扩散包装器
from utils.scheduler import SchedulerInterface  # 导入调度器接口
from typing import List, Optional, Tuple  # 导入常用类型提示
import torch  # 导入 PyTorch
import torch.distributed as dist  # 导入分布式分布式模块
from utils.debug_option import DEBUG, LOG_GPU_MEMORY  # 导入调试开关
from utils.memory import log_gpu_memory  # 导入显存记录工具

class SelfForcingTrainingPipeline:  # 定义自强制训练流水线类
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):  # 初始化函数
        super().__init__()
        self.scheduler = scheduler  # 保存调度器
        self.generator = generator  # 保存生成器模型
        self.denoising_step_list = denoising_step_list  # 保存去噪时间步列表
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # 推理时不需要 0 时间步，将其移除

        # Wan specific hyperparameters  # Wan 模型特有的超参数
        self.num_transformer_blocks = 30  # Transformer 块的数量
        self.frame_seq_length = 1560  # 每帧对应的序列长度（例如 16*16 的潜空间块经过某种排列后的长度）
        self.num_frame_per_block = num_frame_per_block  # 每个时间块包含的帧数
        self.context_noise = context_noise  # 上下文噪声级别
        self.i2v = False  # 是否为图生视频模式，默认为 False

        self.kv_cache1 = None  # 第一组键值对缓存
        self.kv_cache2 = None  # 第二组键值对缓存
        self.crossattn_cache = None  # 交叉注意力缓存
        self.independent_first_frame = independent_first_frame  # 是否开启首帧独立模式
        self.same_step_across_blocks = same_step_across_blocks  # 是否所有块使用相同的时间步
        self.last_step_only = last_step_only  # 是否仅在最后一个时间步进行训练
        # Support local_attn_size as int or list (scheduled by timestep); compute KV cache frames internally
        # 支持整数或列表形式的局部注意力尺寸（可随时间步调度）；内部计算所需的 KV 缓存帧数
        self.local_attn_size = kwargs.get("local_attn_size", -1)
        if not isinstance(self.local_attn_size, int) and hasattr(self.local_attn_size, "__iter__"):
            self.local_attn_size = list(self.local_attn_size)
        if isinstance(self.local_attn_size, (list, tuple)):
            assert len(self.local_attn_size) == len(self.denoising_step_list), (
                f"local_attn_size length ({len(self.local_attn_size)}) must match denoising_step_list length ({len(self.denoising_step_list)})."
            )
            if DEBUG:
                print(f"local_attn_size schedule length: {len(self.local_attn_size)}, denoising steps: {len(self.denoising_step_list)}")
        else:
            if DEBUG:
                print(f"Using static local_attn_size: {self.local_attn_size}")

        # Context used for KV cache calculation  # 用于计算 KV 缓存规模的上下文参数
        num_training_frames: Optional[int] = kwargs.get("num_training_frames", 21)
        slice_last_frames: int = int(kwargs.get("slice_last_frames", 21))

        # Compute KV cache supporting list/int and global attention (-1)  # 解析 KV 缓存所需的总帧数
        def _resolve_kv_frames(local_cfg):
            if isinstance(local_cfg, (list, tuple)):
                base = int(max(local_cfg)) if len(local_cfg) > 0 else -1
                return min(base + slice_last_frames, num_training_frames)
            else:
                base = int(local_cfg)
                return min(base + slice_last_frames, num_training_frames)

        kv_frames = _resolve_kv_frames(self.local_attn_size)
        if DEBUG:
            print(f"[KV policy] local_attn_size={self.local_attn_size} slice_last_frames={slice_last_frames} num_training_frames={num_training_frames} -> kv_frames={kv_frames}")
        self.kv_cache_size = int(kv_frames) * self.frame_seq_length  # 计算最终的 KV 缓存容量（帧数 * 每帧序列长度）

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):  # 生成并同步随机索引列表
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:  # 仅在主进程生成随机索引
            # Generate random indices  # 随机生成去噪深度索引
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:  # 如果仅训练最后一步
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(indices, src=0)  # 将主进程生成的随机索引广播给所有进程
        return indices.tolist()

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        *,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        return_sim_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        Chunk generation method tailored for sequential training
        
        Args:
            noise: noise tensor for a single chunk [batch_size, chunk_frames, C, H, W]
            conditional_dict: dictionary of conditional information
            kv_cache: externally provided KV cache (defaults to self.kv_cache1 if None)
            crossattn_cache: externally provided cross-attention cache (defaults to self.crossattn_cache if None)
            current_start_frame: start frame index of the chunk in the full sequence
            requires_grad: whether gradients are required
            return_sim_step: whether to return simulation step info
            
        Returns:
            output: generated chunk [batch_size, chunk_frames, C, H, W]
            denoised_timestep_from: starting denoise timestep
            denoised_timestep_to: ending denoise timestep
        """  # 专为顺序训练定制的块生成方法，支持缓存。
        batch_size, chunk_frames, num_channels, height, width = noise.shape  # 提取输入张量形状
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):  # 调试输出
            print(f"[SeqTrain-Pipeline] generate_chunk_with_cache: batch_size={batch_size}, chunk_frames={chunk_frames}")
            print(f"[SeqTrain-Pipeline] current_start_frame={current_start_frame}, requires_grad={requires_grad}")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Pipeline: Before chunk generation", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Compute block configuration  # 计算当前块的配置（帧数如何分配到各个子块）
        if not self.independent_first_frame or chunk_frames % self.num_frame_per_block == 0:
            assert chunk_frames % self.num_frame_per_block == 0
            num_blocks = chunk_frames // self.num_frame_per_block
            all_num_frames = [self.num_frame_per_block] * num_blocks
        else:
            # Handle the case of an independent first frame  # 处理包含独立第一帧的情况（如 [1, 4, 4, ...]）
            assert (chunk_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (chunk_frames - 1) // self.num_frame_per_block
            all_num_frames = [1] + [self.num_frame_per_block] * num_blocks
            
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] Block config: num_blocks={num_blocks}, all_num_frames={all_num_frames}")
            print(f"[SeqTrain-Pipeline] independent_first_frame={self.independent_first_frame}")
            
        # Prepare output tensor  # 准备输出张量空间
        output = torch.zeros_like(noise)
        
        # Randomly select denoising steps (synced across ranks)  # 随机选择去噪深度索引（并在各进程间同步）
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):  # 调试：打印去噪步数和退出标志
            print(f"[SeqTrain-Pipeline] Denoising steps: {num_denoising_steps}, exit_flags: {exit_flags}")
        
        # Determine gradient-enabled range — disable everywhere when requires_grad=False
        # 确定梯度开启范围 —— 当 requires_grad 为 False 时，在任何地方都禁用梯度
        if not requires_grad:
            start_gradient_frame_index = chunk_frames  # 超出范围：各处均无梯度
        else:
            start_gradient_frame_index = 0
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):  # 调试：打印梯度起始帧索引
            print(f"[SeqTrain-Pipeline] start_gradient_frame_index={start_gradient_frame_index}")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Pipeline: Before block generation loop", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Generate block by block  # 逐块生成
        local_start_frame = 0
        # If static local_attn_size, set it on the model before the step loop
        # 如果是静态局部注意力尺寸，则在步骤循环前设置到模型上
        if not (isinstance(self.local_attn_size, (list, tuple)) or (hasattr(self.local_attn_size, "__iter__") and not isinstance(self.local_attn_size, (str, bytes)))):
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))
        for block_index, current_num_frames in enumerate(all_num_frames):  # 遍历每个时间块
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Pipeline] Processing block {block_index}: frames {local_start_frame}-{local_start_frame + current_num_frames}")
            
            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and block_index == 0:
                log_gpu_memory(f"SeqTrain-Pipeline: Before first block generation", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
                
            noisy_input = noise[:, local_start_frame:local_start_frame + current_num_frames]  # 获取当前块的带噪输入
            
            # Spatial denoising loop  # 空间去噪循环（迭代加噪/去噪过程）
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                # If scheduled, set local_attn_size dynamically per timestep
                # 如果是调度模式，按每个时间步动态设置局部注意力尺寸
                if isinstance(self.local_attn_size, (list, tuple)) or (hasattr(self.local_attn_size, "__iter__") and not isinstance(self.local_attn_size, (str, bytes))):
                    self.generator.model.local_attn_size = int(self.local_attn_size[step_idx])
                    if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                        print(f"[denoise step {step_idx}] timestep={float(current_timestep)} local_attn_size={self.generator.model.local_attn_size}")
                    self._set_all_modules_max_attention_size(int(self.local_attn_size[step_idx]))
                exit_flag = (
                    step_idx == exit_flags[0]
                    if self.same_step_across_blocks
                    else step_idx == exit_flags[block_index]
                )  # 判断当前步是否为该块的最终输出步
                
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep  # 构造当前的时间步张量
                
                if not exit_flag:
                    # Intermediate steps: no gradients  # 中间步骤：不保留梯度，仅进行前向传播
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Pipeline] Block {block_index} intermediate steps (no grad)")
                        
                    with torch.no_grad():
                        _, denoised_pred = self.generator(  # 运行生成器进行一步预测
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )
                        
                        # Add noise for the next step  # 为下一步添加噪声，构成后向模拟轨迹
                        if step_idx < len(self.denoising_step_list) - 1:
                            next_timestep = self.denoising_step_list[step_idx + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                                ),
                            ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Final step may require gradients  # 最终出口步：根据需要决定是否开启梯度
                    enable_grad = local_start_frame >= start_gradient_frame_index
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Pipeline] Block {block_index} final step: enable_grad={enable_grad}")
                    
                    context_manager = torch.enable_grad() if enable_grad else torch.no_grad()
                    with context_manager:
                        _, denoised_pred = self.generator(  # 生成最终结果
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )
                    break  # 达到出口步，跳出当前块的迭代
            
            # Record output  # 记录当前块生成的潜变量到输出张量
            output[:, local_start_frame:local_start_frame + current_num_frames] = denoised_pred
            
            # Update cache with context noise  # 使用上下文噪声更新 KV 缓存（自强制训练的关键，注入一定随机性）
            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noisy = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])
            
            if DEBUG and block_index == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Pipeline] Updating cache with context_noise={self.context_noise}")
            
            with torch.no_grad():
                self.generator(  # 将带噪预测结果再次输入以更新 KV 缓存（但不保留结果，仅利用其副作用）
                    noisy_image_or_video=context_noisy,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                )
            
            local_start_frame += current_num_frames  # 更新本块内起始帧位置
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Pipeline: After all blocks generated", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Compute returned timestep information  # 计算并返回对应的去噪起止时间步（用于计算 Loss 的权重）
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()
        
        if return_sim_step:  # 如果需要，额外返回模拟进行的步数等级
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1
        
        return output, denoised_timestep_from, denoised_timestep_to

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            slice_last_frames: int = 21,
            **conditional_dict
    ):  # 带轨迹记录的完整推理过程实现
            slice_last_frames: int = 21,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape  # 提取输入噪声形状
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            # 如果首帧独立且已提供，则噪声帧数仍应为块大小的倍数
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            # 使用包含独立首帧的基础配置生成无图像条件的视频
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0  # 初始潜变量的帧数
        num_output_frames = num_frames + num_input_frames  # 计算总输出帧数（噪声帧 + 初始帧）
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )  # 创建输出张量

        # Step 1: Initialize KV cache to all zeros  # 步骤 1: 将 KV 缓存初始化为全零
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        # Step 2: Cache context feature  # 步骤 2: 缓存上下文特征（处理初始帧）
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent  # 将初始潜变量填入输出开头
            with torch.no_grad():
                self.generator(  # 运行生成器以填充 KV 缓存
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    clip_fea=conditional_dict.get("clip_fea"),
                    y=[u[:, :1] for u in conditional_dict.get("y")] if conditional_dict.get("y") is not None else None
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop  # 步骤 3: 时间维度去噪循环（逐块推进）
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames  # 无初始帧时的首帧独立处理
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - slice_last_frames  # 计算开始保留梯度的帧索引

        grad_enable_mask = torch.zeros((batch_size, sum(all_num_frames)), dtype=torch.bool)
        # If static local_attn_size, set it first  # 静态局部注意力设置
        if not isinstance(self.local_attn_size, (list, tuple)):
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))
        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):  # 遍历每个时间块
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]  # 获取本块噪声

            # Step 3.1: Spatial denoising loop  # 步骤 3.1: 空间去噪循环（块内迭代）
            for index, current_timestep in enumerate(self.denoising_step_list):
                # If scheduled, set local_attn_size dynamically per timestep
                # 如果是调度模式，按当前去噪步动态设置局部注意力尺寸
                if isinstance(self.local_attn_size, (list, tuple)):
                    self.generator.model.local_attn_size = int(self.local_attn_size[index])
                    if not dist.is_initialized() or dist.get_rank() == 0 and DEBUG:
                        print(f"[denoise step {index}] timestep={float(current_timestep)} local_attn_size={self.generator.model.local_attn_size}")
                    self._set_all_modules_max_attention_size(int(self.local_attn_size[index]))
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # 仅在随机选定的步数进行反向传播（跨进程一致）
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep  # 构造时间步张量
                if DEBUG and dist.get_rank() == 0:
                    print(f"rank {dist.get_rank()}, current_start_frame: {current_start_frame}, current_num_frames: {current_num_frames}, current_timestep: {current_timestep}")
                if not exit_flag:
                    with torch.no_grad():  # 中间步不需要梯度
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            clip_fea=conditional_dict.get("clip_fea"),
                            y=[u[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames] for u in conditional_dict.get("y")] if conditional_dict.get("y") is not None else None
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(  # 向预测结果加噪以进行下一步迭代
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output  # 出口步，获取最终预测结果
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = False
                        _, denoised_pred = self.generator(  # 无梯度生成（早期帧）
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            clip_fea=conditional_dict.get("clip_fea"),
                            y=[u[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames] for u in conditional_dict.get("y")] if conditional_dict.get("y") is not None else None
                        )
                    else:
                        # print(f"enable grad: {current_start_frame}")
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = True
                        _, denoised_pred = self.generator(  # 潜在带梯度的生成（尾部蒸馏帧）
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            clip_fea=conditional_dict.get("clip_fea"),
                            y=[u[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames] for u in conditional_dict.get("y")] if conditional_dict.get("y") is not None else None
                        )
                    break  # 达到出口步，完成当前块推理
            
            # Step 3.2: record the model's output  # 步骤 3.2: 记录模型的输出结果
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache  # 步骤 3.3: 使用 context_noise 重新运行以更新 KV 缓存
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise  # 添加上下文侧重噪声
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(  # 仅用于刷新缓存
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    clip_fea=conditional_dict.get("clip_fea"),
                    y=[u[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames] for u in conditional_dict.get("y")] if conditional_dict.get("y") is not None else None
                )

            # Step 3.4: update the start and end frame indices  # 步骤 3.4: 更新起始帧和结束帧索引
            current_start_frame += current_num_frames

        if dist.get_rank() == 0 and DEBUG:
            print(f"grad_enable_mask: {grad_enable_mask[0, :]}")
            
        # Step 3.5: Return the denoised timestep  # 步骤 3.5: 返回对应的去噪时间步范围
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:  # 如果需要，返回模拟步数等级
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):  # 初始化 KV 缓存
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """  # 为 Wan 模型初始化每个 GPU 独立的 KV 缓存。
        kv_cache1 = []
        if DEBUG:
            print(f"rank {dist.get_rank()} initialize kv cache with batch_size: {batch_size}, kv_cache_size: {self.kv_cache_size}")
        for _ in range(self.num_transformer_blocks):  # 为每个 Transformer 块创建缓存槽
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),  # Key 缓存张量
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),  # Value 缓存张量
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),  # 全局结束位置索引
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)  # 局部结束位置索引
            })

        self.kv_cache1 = kv_cache1  # 存储缓存列表

    def _initialize_crossattn_cache(self, batch_size, dtype, device):  # 初始化交叉注意力缓存
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """  # 交叉注意力（针对 Prompt 条件）在同一次推理中是静态的，进行缓存可加速
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False  # 标识位：是否已根据当前 Prompt 初始化
            })
        self.crossattn_cache = crossattn_cache

    def clear_kv_cache(self):  # 清空 KV 缓存
        """
        Zero out all tensors in KV cache and cross-attention cache instead of setting them to None.
        This preserves memory allocation while clearing old information, avoiding reallocation overhead.
        """  # 将 KV 缓存和交叉注意力缓存张量清零，而非设为 None，以保留显存空间分配，避免频繁重分配开销。

        # Clear KV cache  # 清空 KV 缓存
        if getattr(self, "kv_cache1", None) is not None:
            for blk in self.kv_cache1:
                blk["k"].zero_()
                blk["v"].zero_()
                if "global_end_index" in blk:
                    blk["global_end_index"].zero_()
                if "local_end_index" in blk:
                    blk["local_end_index"].zero_()

        # Clear cross-attention cache  # 清空交叉注意力缓存
        if getattr(self, "crossattn_cache", None) is not None:
            for blk in self.crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):  # 设置注意力机制的最大有效范围
        """
        Set a unified upper bound for all submodules that contain the max_attention_size attribute.
        local_attn_size_value == -1 indicates global attention (use Wan's default token limit 32760).
        Otherwise set to local_attn_size_value * frame_seq_length.
        """  # 为所有包含 max_attention_size 属性的子模块设置统一的上限。-1 代表全局注意力。
        if isinstance(local_attn_size_value, (list, tuple)):
            raise ValueError("_set_all_modules_max_attention_size expects an int, got list/tuple.")

        if int(local_attn_size_value) == -1:  # 全局模式：使用默认的大 token 限制
            target_size = 32760
            policy = "global"
        else:  # 局部模式：计算总的注意力窗口 token 长度
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        # Root module  # 根模型设置
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                _ = getattr(self.generator.model, "max_attention_size")
            except Exception:
                pass
            setattr(self.generator.model, "max_attention_size", target_size)

        # Child modules  # 递归设置所有子模块（如 AttentionBlock）
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    setattr(module, "max_attention_size", target_size)
                except Exception:
                    pass
