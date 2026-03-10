# Adopted from https://github.com/guandeh17/Self-Forcing  # 适配自 Self-Forcing 开源项目
# SPDX-License-Identifier: Apache-2.0  # 许可证信息
import torch.nn.functional as F  # 导入神经网络常用算子库
from typing import Optional, Tuple  # 导入类型提示常用组件
import torch  # 导入主 PyTorch 库
import time  # 导入时间库

from model.base import SelfForcingModel  # 从基础模型库导入自强制模型基类
from utils.memory import log_gpu_memory  # 导入用于显存追踪的实用工具
import torch.distributed as dist  # 导入分布式训练支持包
from utils.debug_option import DEBUG, LOG_GPU_MEMORY  # 从调试选项中导入调试标志

class DMD(SelfForcingModel):  # 定义 DMD (分布匹配蒸馏) 模型类，继承自 SelfForcingModel
    def __init__(self, args, device):  # 类初始化函数
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """  # 初始化 DMD 模块，该类实现了在一次前向传播中同时计算生成器和判别器(fake score)的损失
        super().__init__(args, device)  # 执行父类初始化逻辑
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)  # 获取配置中的每块视频帧数
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)  # 是否在不同块中使用相同的去噪步长
        self.min_num_training_frames = getattr(args, "min_num_training_frames", 21)  # 训练时所需的最少总帧数
        self.num_training_frames = getattr(args, "num_training_frames", 21)  # 训练时的目标总帧数

        if self.num_frame_per_block > 1:  # 若是视频模式（即块帧数大于 1）
            self.generator.model.num_frame_per_block = self.num_frame_per_block  # 同步设置生成器内部的块帧数

        self.independent_first_frame = getattr(args, "independent_first_frame", False)  # 是否将首帧作为独立条件而不依赖以往缓存
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True  # 同步设置生成器的独立首帧标志
        if args.gradient_checkpointing:  # 如果开启了梯度检查点技术以节约显存
            self.generator.enable_gradient_checkpointing()  # 为生成器开启该技术
            self.fake_score.enable_gradient_checkpointing()  # 为判别器(fake_score)开启该技术

        # this will be init later with fsdp-wrapped modules  # 该项推后交由后续经 FSDP 包装过的模块进行正式初始化
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters  # 步骤 2: 初始化 DMD 算法的所有超参数
        self.num_train_timestep = args.num_train_timestep  # 提取总训练步数
        self.min_step = int(0.02 * self.num_train_timestep)  # 计算取样的最小时间步窗口（总数的 2%）
        self.max_step = int(0.98 * self.num_train_timestep)  # 计算取样的最大时间步窗口（总数的 98%）
        if hasattr(args, "real_guidance_scale"):  # 如果配置文件中对真实分布和判别分布提供了差异化的引导系数
            self.real_guidance_scale = args.real_guidance_scale  # 真实样本的分类器自由引导 (CFG) 系数
            self.fake_guidance_scale = args.fake_guidance_scale  # 生成样本的 CFG 系数
        else:
            self.real_guidance_scale = args.guidance_scale  # 否则统一使用通用引导系数
            self.fake_guidance_scale = 0.0  # 默认生成样本不使用额外引导系数
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)  # 获取时间步偏移系数
        self.ts_schedule = getattr(args, "ts_schedule", True)  # 是否开启时间步调度机制
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)  # 调度机制是否始终锁定在最大值
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)  # 开始评分的最小起始时间步限制

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:  # 如果调度器包含累积 alpha 值
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)  # 将其迁移至训练设备上
        else:
            self.scheduler.alphas_cumprod = None  # 否则设为空

    def _compute_kl_grad(  # 计算 KL 散度的梯度分布（蒸馏核心：对应 DMD 论文等式 7）
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:  # 参数包含加噪样本、预估的干净样本、时间步以及控制字典
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score  # 步骤 1: 计算假分布（生成分布）的评分 (Score)
        clip_fea = conditional_dict.get("clip_fea")  # 提取 CLIP 图像特征
        y = conditional_dict.get("y")  # 提取 VAE 的潜变量表征
        _, pred_fake_image_cond = self.fake_score(  # 使用正在训练的判别器对带条件的加噪样本进行去噪预测
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clip_fea=clip_fea,
            y=y
        )

        if self.fake_guidance_scale != 0.0:  # 如果生成分布开启了引导机制
            _, pred_fake_image_uncond = self.fake_score(  # 同样对无条件样本进行去噪预测以实现 CFG
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                clip_fea=clip_fea,
                y=y
            )
            pred_fake_image = pred_fake_image_cond + (  # 利用 CFG 公式组合出最终预测的假样本去噪结果
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:  # 若关闭引导，则直接使用带条件预测结果
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score  # 步骤 2: 计算真实分布的评分（由 Teacher/RealScore 模型引导）
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(  # 由老师模型预测带条件样本的去噪结果
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clip_fea=clip_fea,
            y=y
        )

        _, pred_real_image_uncond = self.real_score(  # 由老师模型预测无条件样本的去噪结果
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            clip_fea=clip_fea,
            y=y
        )

        pred_real_image = pred_real_image_cond + (  # 组合出目标真实样本去噪分布的方向 (CFG)
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).  # 步骤 3: 两者相减得出 DMD 梯度（差异指导生成器向真实分布移动）
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:  # 若开启归一化（对应 DMD 论文等式 8）
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)  # 计算真实预测偏移量
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)  # 在视频维度上求均值作为归一化因数
            grad = grad / normalizer  # 对梯度进行归一化，防止特定步长下梯度过大或过小
        grad = torch.nan_to_num(grad)  # 将无效数值 (NaN) 转换为数字，防止训练死锁

        return grad, {  # 返回计算出的梯度以及用于日志记录的各项元指标
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),  # 记录梯度的绝对值范数
            "timestep": timestep.detach()  # 记录当前采样的时间步
        }
    def compute_distribution_matching_loss(  # 计算分布匹配损失函数
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video  # 记录输入的原始潜变量

        batch_size, num_frame = image_or_video.shape[:2]  # 提取批大小和帧数

        with torch.no_grad():  # 梯度计算过程的部分中间量不需要保留梯度
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            # 步骤 1: 根据给定的调度方案随机采样时间步及对应的噪声
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep  # 确定采样时间步的下限
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep  # 确定采样时间步的上限
            timestep = self._get_timestep(  # 获取训练用的随机时间步
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:  # 如果开启了时间步偏移（通常用于调整训练难度分布）
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)  # 将时间步限制在安全合法范围内

            noise = torch.randn_like(image_or_video)  # 生成与潜变量同形的随机高斯噪声
            noisy_latent = self.scheduler.add_noise(  # 向潜变量中注入噪声以构建加噪样本
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))  # 分离梯度并还原维度形状

            # Step 2: Compute the KL grad  # 步骤 2: 计算 KL 散度下降的梯度方向
            grad, dmd_log_dict = self._compute_kl_grad(  # 调用内部方法得到蒸馏所需的指导梯度
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        if gradient_mask is not None:  # 如果提供了掩码（通常用于因果生成中仅对比新生成的块）
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(  # 计算带掩码的均方误差损失
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:  # 若无掩码
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(  # 直接计算全局均方误差作为 DMD 训练损失
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict  # 返回标量损失及日志字典

    def generator_loss(  # 计算生成器的蒸馏损失主入口
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:  # 调试：记录展开生成前显存
            log_gpu_memory(f"Generator loss: Before generator unroll", device=self.device, rank=dist.get_rank())
        # Step 1: Unroll generator to obtain fake videos  # 步骤 1: 展开生成器以获取生成的视频样本
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)  # 获取切片帧数
        _t_gen_start = time.time()  # 记录开始生成的时间
        if DEBUG and dist.get_rank() == 0:
            print(f"generator_rollout")
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(  # 执行生成器前馈
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            slice_last_frames=slice_last_frames
        )
        if dist.get_rank() == 0 and DEBUG:  # 调试：打印生成信息的统计情况
            print(f"pred_image: {pred_image.shape}")
            if gradient_mask is not None:   
                print(f"gradient_mask: {gradient_mask[0, :, 0, 0, 0]}")
            else:
                print(f"gradient_mask: None")
        gen_time = time.time() - _t_gen_start  # 计算生成耗时
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:  # 调试：记录生成后显存
            log_gpu_memory(f"Generator loss: After generator unroll", device=self.device, rank=dist.get_rank())
        # Step 2: Compute the DMD loss  # 步骤 2: 计算 DMD 训练损失值
        _t_loss_start = time.time()
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(  # 对生成的样本计算分布匹配损失
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:  # 调试：记录损失计算后显存
            log_gpu_memory(f"Generator loss: After compute_distribution_matching_loss", device=self.device, rank=dist.get_rank())
        try:
            loss_val = dmd_loss.item()
        except Exception:
            loss_val = float('nan')
        loss_time = time.time() - _t_loss_start  # 计算损失函数计算耗时
        # print(f"[GeneratorLoss] loss {loss_val} | gen_time {gen_time:.3f}s | loss_time {loss_time:.3f}s")

        dmd_log_dict.update({  # 更新日志字典中的计时信息
            "gen_time": gen_time,
            "loss_time": loss_time
        })

        return dmd_loss, dmd_log_dict  # 返回最终生成器损失及记录

    def critic_loss(  # 计算判别器（Critic）的训练损失
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:  # 调试：记录判别器生成前显存
            log_gpu_memory(f"Critic loss: Before generator unroll", device=self.device, rank=dist.get_rank())
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)  # 获取切片帧数
        # Step 1: Run generator on backward simulated noisy input
        # 步骤 1: 在通过后向模拟得到的加噪输入上运行生成器，得到“假”样本
        _t_gen_start = time.time()
        with torch.no_grad():  # 此时生成器作为样本生成源，不需要保留梯度
            if DEBUG and dist.get_rank() == 0:
                print(f"critic_rollout")
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(  # 执行生成
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                slice_last_frames=slice_last_frames
            )
        if dist.get_rank() == 0 and DEBUG:
            print(f"pred_image: {generated_image.shape}")
        gen_time = time.time() - _t_gen_start  # 记录生成耗时
        batch_size, num_frame = generated_image.shape[:2]  # 获取批次和帧数信息
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:  # 调试：记录生成后显存
            log_gpu_memory(f"Critic loss: After generator unroll", device=self.device, rank=dist.get_rank())
        _t_loss_start = time.time()

        # Step 2: Compute the fake prediction  # 步骤 2: 计算判别器对生成的假样本的预测
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(  # 为判别器训练随机采样时间步
            min_timestep,
            max_timestep,
            batch_size,
            num_frame,
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:  # 同样应用时间步偏移逻辑
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)  # 将时间步限制在安全范围内

        critic_noise = torch.randn_like(generated_image)  # 为生成的假样本产生训练噪声
        noisy_generated_image = self.scheduler.add_noise(  # 向生成的假样本注入噪声，模拟扩散过程中间态
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))

        _, pred_fake_image = self.fake_score(  # 让判别器预测当前带噪生成的“原生”图像（即回归 x0）
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
            clip_fea=conditional_dict.get("clip_fea"),
            y=conditional_dict.get("y")
        )

        # Step 3: Compute the denoising loss for the fake critic  # 步骤 3: 计算判别器的去噪损失
        if self.args.denoising_loss_type == "flow":  # 若损失类型为 Flow Matching
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(  # 将预测的 x0 转换为 Flow 预测结果
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:  # 若为标准去噪损失，则将 x0 转化为预测的噪声
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))

        denoising_loss = self.denoising_loss_func(  # 计算去噪均方误差损失，以此训练判别器捕捉生成样本的分布
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        try:
            loss_val = denoising_loss.item()
        except Exception:
            loss_val = float('nan')
        loss_time = time.time() - _t_loss_start  # 记录损失计算耗时
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:  # 调试：记录损失计算后显存
            log_gpu_memory(f"Critic loss: After denoising loss", device=self.device, rank=dist.get_rank())
        # print(f"[CriticLoss] loss {loss_val} | gen_time {gen_time:.3f}s | loss_time {loss_time:.3f}s")


        # Step 5: Debugging Log  # 步骤 5: 整理调试及记录日志
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach(),
            "gen_time": gen_time,
            "loss_time": loss_time
        }

        return denoising_loss, critic_log_dict  # 返回判别器损失以及对应日志
