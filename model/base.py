# Adopted from https://github.com/guandeh17/Self-Forcing  # 采用自 Self-Forcing 开源项目
# SPDX-License-Identifier: Apache-2.0  # 采用 Apache 2.0 开源协议
from typing import Tuple  # 导入元组类型提示
from einops import rearrange  # 导入 einops 用于张量维度变换
from torch import nn  # 导入 PyTorch 的 nn 模块
import torch.distributed as dist  # 导入分布式分布式模块
import torch  # 导入 PyTorch 核心库

from pipeline import SelfForcingTrainingPipeline  # 导入自强制训练流水线
from utils.loss import get_denoising_loss  # 导入获取去噪损失函数的工具
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, WanCLIPEncoder  # 导入 Wan 模型相关的包装器

from utils.debug_option import DEBUG  # 导入调试开关

class BaseModel(nn.Module):  # 定义基础模型类，继承自 nn.Module
    def __init__(self, args, device):  # 初始化函数
        super().__init__()  # 调用父类初始化
        self.i2v = getattr(args, "i2v", False)  # 获取是否为图生视频 (I2V) 模式
        self._initialize_models(args, device)  # 调用内部方法初始化子模型

        self.device = device  # 记录设备信息
        self.args = args  # 记录配置参数
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32  # 设置数据类型
        if hasattr(args, "denoising_step_list"):  # 如果配置了去噪步骤列表
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)  # 转化为张量
            if args.warp_denoising_step:  # 如果开启了时间步扭曲 (warp)
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))  # 合并时间步并补 0
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]  # 映射到对应的调度器时间步

    def _initialize_models(self, args, device):  # 初始化各子模块
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")  # 获取真实分数模型名称
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")  # 获取假分数模型名称
        self.generator_name = getattr(args, "generator_name", "Wan2.1-T2V-1.3B")  # 获取生成器模型名称
        self.local_attn_size = getattr(args, "model_kwargs", {}).get("local_attn_size", -1)  # 获取局部注意力窗口大小
        self.generator = WanDiffusionWrapper(  # 初始化生成器包装器
            model_name=self.generator_name,
            **getattr(args, "model_kwargs", {}), 
            is_causal=True  # 启用因果模型模式
        )
        self.generator.model.requires_grad_(True)  # 生成器需要更新参数

        self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)  # 初始化真实分数预测器 (Teacher)
        self.real_score.model.requires_grad_(False)  # 真实分数模型（老师）固定参数

        self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, is_causal=False)  # 初始化假分数预测器 (Critic)
        self.fake_score.model.requires_grad_(True)  # 假分数模型（学生判别器）需要更新参数

        self.text_encoder = WanTextEncoder(model_name=self.generator_name)  # 初始化文本编码器
        self.text_encoder.requires_grad_(False)  # 文本编码器通常固定

        self.vae = WanVAEWrapper(model_name=self.generator_name)  # 初始化 VAE 包装器
        self.vae.requires_grad_(False)  # VAE 为预训练好的，无需训练

        # Initialize image encoder for I2V (Image-to-Video)  # 为 I2V 模式初始化图像编码器
        if self.i2v:
            self.image_encoder = WanCLIPEncoder(model_name=self.generator_name)  # 实例化 CLIP 图像编码器
            self.image_encoder.requires_grad_(False)  # 图像编码器固定

        self.scheduler = self.generator.get_scheduler()  # 从生成器获取扩散调度器 (Scheduler)
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)  # 将时间步张量移动到当前设备

    def _get_timestep(  # 获取时间步张量的方法
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """  # 根据任务类型随机生成时间步张量，形状为 [batch_size, num_frame]
        if uniform_timestep:  # 如果要求所有帧的时间步一致
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)  # 在批内采样后重复到所有帧
            return timestep
        else:  # 如果允许不同帧有不同时间步（通常对每一块 block 进行统一）
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block  # 强制使每一个时间块内部的噪声水平一致
            if self.independent_first_frame:  # 如果开启了独立首帧模式
                # the first frame is always kept the same  # 第一帧的时间步保持采样结果
                timestep_from_second = timestep[:, 1:]  # 取出从第二帧开始的部分
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)  # 按照块大小变形
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]  # 将块内的后续帧时间步同步为块内首帧
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)  # 还原平面形状
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)  # 重新拼接首帧和处理后的后续帧
            else:  # 正常的块对齐逻辑
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)  # 将总帧数按块大小切分
                timestep[:, :, 1:] = timestep[:, :, 0:1]  # 块内时间步取对齐
                timestep = timestep.reshape(timestep.shape[0], -1)  # 还原形状
            return timestep
class SelfForcingModel(BaseModel):  # 定义自强制 (Self-Forcing) 模型类，继承自 BaseModel
    def __init__(self, args, device):  # 初始化函数
        super().__init__(args, device)  # 调用基类初始化
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()  # 根据配置获取对应的去噪损失计算函数

    def _run_generator(  # 运行生成器以产生样本的核心逻辑
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
        slice_last_frames: int = 21,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """  # 通过后向模拟从噪声中产生生成器的输入，然后运行生成器。
        # Step 1: Sample noise and backward simulate the generator's input  # 步骤 1: 采样噪声并对生成器输入进行后向轨迹模拟
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"  # 确保开启了后向模拟标志
        if initial_latent is not None:  # 如果提供了初始潜变量（如 I2V 的首帧或接续帧）
            conditional_dict["initial_latent"] = initial_latent  # 将其加入条件字典
        if self.args.i2v:  # 如果是 I2V 模式
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]  # 噪声帧数减 1（首帧已知）
        else:  # T2V 模式
            noise_shape = image_or_video_shape.copy()  # 全帧均为噪声

        # During training, the number of generated frames should be uniformly sampled from  # 训练期间生成的帧数应在 [min, max] 之间均匀采样
        # [min_num_frames, self.num_training_frames], but still being a multiple of self.num_frame_per_block.
        # If `min_num_frames` is not provided, we fallback to the original default behaviour.  # 且必须是块大小的整数倍
        min_num_frames = (self.min_num_training_frames - 1) if self.args.independent_first_frame else self.min_num_training_frames
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0  # 校验最大帧数是否对齐块
        assert min_num_frames % self.num_frame_per_block == 0  # 校验最小帧数是否对齐块
        max_num_blocks = max_num_frames // self.num_frame_per_block  # 计算最大块数
        min_num_blocks = min_num_frames // self.num_frame_per_block  # 计算最小块数
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)  # 随机决定本次生成的块数
        dist.broadcast(num_generated_blocks, src=0)  # 在分布式进程间同步块数，确保所有 GPU 生成相同长度的序列
        num_generated_blocks = num_generated_blocks.item()  # 转为标量
        num_generated_frames = num_generated_blocks * self.num_frame_per_block  # 最终生成的帧数（块对齐后）
        if dist.get_rank() == 0 and DEBUG:  # 调试：打印生成的帧数
            print(f"num_generated_frames: {num_generated_frames}")
        if self.args.independent_first_frame and initial_latent is None:  # 对于独立首帧但无初始潜变量情况（如 T2V 开启该项）
            num_generated_frames += 1  # 总长度补回首帧
            min_num_frames += 1
        # Sync num_generated_frames across all processes  # 同步噪声形状中的帧数维度
        noise_shape[1] = num_generated_frames

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(  # 调用一致性后向模拟获取生成结果
            noise=torch.randn(noise_shape,
                               device=self.device, dtype=self.dtype),  # 临时采样高斯噪声
            slice_last_frames=slice_last_frames,
            **conditional_dict,
        )
        # Decide whether to slice based on `slice_last_frames`; when `slice_last_frames == -1`, keep all frames
        # 根据 slice_last_frames 决定是否切片；若为 -1 则保留全部生成的轨迹帧
        if slice_last_frames != -1 and pred_image_or_video.shape[1] > slice_last_frames:
            with torch.no_grad():  # 潜变量编解码过程不需要梯度
                # Re-encode: take all frames before the last (slice_last_frames - 1) frames for pixel decoding
                # 重新编码：取出最后一部分之前的全部帧进行像素解码（用于获取更稳定的潜变量表示）
                if slice_last_frames > 1:
                    latent_to_decode = pred_image_or_video[:, :-(slice_last_frames - 1), ...]  # 截取前半段
                else:
                    latent_to_decode = pred_image_or_video  # 全量解码
                # Decode to video  # 解码回像素空间
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)  # 取最后一帧像素
                frame = rearrange(frame, "b t c h w -> b c t h w")  # 调整维度以适配 VAE 编码器
                # Encode frame to get image latent  # 将像素帧重新编码回潜空间，以获取更精确的 conditioning
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            if slice_last_frames > 1:
                last_frames = pred_image_or_video[:, -(slice_last_frames - 1):, ...]  # 取出未被替换的后续部分帧
                pred_image_or_video_sliced = torch.cat([image_latent, last_frames], dim=1)  # 将重编码帧和后续帧拼接
            else:
                pred_image_or_video_sliced = image_latent  # 仅剩当前单帧
        else:  # 不做切片
            pred_image_or_video_sliced = pred_image_or_video

        if num_generated_frames != min_num_frames:  # 如果生成的帧数长于最小要求（即包含历史缓存对应部分）
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            # 目前不对第一个块应用梯度，因为它含有作为条件的图像潜变量（历史信息）
            gradient_mask = torch.ones_like(pred_image_or_video_sliced, dtype=torch.bool)  # 初始化全 1 掩码
            if self.args.independent_first_frame:  # 如果是独立首帧
                gradient_mask[:, :1] = False  # 首帧遮蔽梯度
            else:  # 否则遮蔽第一个时间块
                gradient_mask[:, :self.num_frame_per_block] = False
        else:  # 若是初始全量生成，则无需梯度掩码
            gradient_mask = None

        pred_image_or_video_sliced = pred_image_or_video_sliced.to(self.dtype)  # 确保精度一致
        return pred_image_or_video_sliced, gradient_mask, denoised_timestep_from, denoised_timestep_to  # 返回生成样本、掩码及时间步范围

    def _consistency_backward_simulation(  # 一致性采样后向轨迹模拟的具体实现
        self,
        noise: torch.Tensor,
        slice_last_frames: int = 21,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """  # 模拟生成器从噪声到样本的输入轨迹，以消除训推不一致。基于 DMD2 论文 4.5 节。
        if self.inference_pipeline is None:  # 如果推理管线尚未创建
            self._initialize_inference_pipeline()  # 执行懒加载初始化

        return self.inference_pipeline.inference_with_trajectory(  # 调用管线方法执行带轨迹记录的推理
            noise=noise, **conditional_dict, slice_last_frames=slice_last_frames
        )

    def _initialize_inference_pipeline(self):  # 初始化用于训练中后向模拟的推理管线
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """  # 在首次后向模拟时进行懒加载初始化，将 FSDP 包装过的模块传入管线以节省显存。
        local_attn_size = getattr(self.args, "model_kwargs", {}).get("local_attn_size", -1)  # 获取局部注意力参数
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)  # 获取切片帧数参数
        # do not use self.num_training_frames, because it is changed by generator_loss and critic_loss
        # 不使用由 generator_loss 修改过的动态值，而是从原始参数中读取固定的训练最大帧数
        num_training_frames = getattr(self.args, "num_training_frames")
        self.inference_pipeline = SelfForcingTrainingPipeline(  # 实例化自强制训练推理管线
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=num_training_frames,
            context_noise=self.args.context_noise,
            local_attn_size=local_attn_size,
            slice_last_frames=slice_last_frames,
            num_training_frames=num_training_frames,
        )
