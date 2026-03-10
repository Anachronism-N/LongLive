# Adopted from https://github.com/guandeh17/Self-Forcing  # 采用自 Self-Forcing 开源项目
# SPDX-License-Identifier: Apache-2.0  # 采用 Apache 2.0 开源协议
import os  # 导入操作系统相关模块
import types  # 导入类型相关模块
from typing import List, Optional  # 导入常用类型提示
import torch  # 导入 PyTorch
from torch import nn  # 导入神经网络模块

from utils.scheduler import SchedulerInterface, FlowMatchScheduler  # 导入调度器接口及 FlowMatch 调度器
from wan.modules.tokenizers import HuggingfaceTokenizer  # 导入 Wan 模型专用的分词器
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock  # 导入 Wan 核心模型组件
from wan.modules.vae import _video_vae  # 导入视频 VAE 模型加载函数
from wan.modules.t5 import umt5_xxl  # 导入 T5 文本编码器
from wan.modules.clip import CLIPModel  # 导入 CLIP 视觉模型
from wan.modules.causal_model import CausalWanModel  # 导入因果 Wan 模型
from wan.modules.causal_model_infinity import CausalWanModel as CausalWanModelInfinity  # 导入支持无限长度的因果 Wan 模型

class WanTextEncoder(torch.nn.Module):  # 定义 Wan 文本编码器封装类
    def __init__(self, model_name="Wan2.1-T2V-1.3B") -> None:  # 初始化函数，默认为 1.3B 模型
        super().__init__()
        self.model_name = model_name  # 保存模型名称

        self.text_encoder = umt5_xxl(  # 初始化 UMT5 XXL 编码器
            encoder_only=True,  # 仅使用编码器部分
            return_tokenizer=False,  # 不在此时返回分词器
            dtype=torch.float32,  # 使用单精度浮点数
            device=torch.device('cpu')  # 初始加载到 CPU
        ).eval().requires_grad_(False)  # 设为评估模式并禁用梯度计算
        self.text_encoder.load_state_dict(  # 加载预训练权重
            torch.load(f"wan_models/{self.model_name}/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )
        
        # Move text encoder to GPU if available  # 如果 GPU strike，则将文本编码器移至 GPU
        if torch.cuda.is_available():
            self.text_encoder = self.text_encoder.cuda()

        self.tokenizer = HuggingfaceTokenizer(  # 初始化分词器，指定路径、长度及清洗策略
            name=f"wan_models/{self.model_name}/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):  # 获取设备属性
        # Assume we are always on GPU  # 假设我们总是在 GPU 上运行
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:  # 前向传播函数
        ids, mask = self.tokenizer(  # 对文本提示进行分词并获取掩码
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)  # 将 ID 移至 GPU
        mask = mask.to(self.device)  # 将掩码移至 GPU
        seq_lens = mask.gt(0).sum(dim=1).long()  # 计算每个提示的有效序列长度
        context = self.text_encoder(ids, mask)  # 执行文本编码
        # ids = ids.to(torch.device('cpu'))
        # mask = mask.to(torch.device('cpu'))
        for u, v in zip(context, seq_lens):  # 遍历编码结果
            u[v:] = 0.0  # 将填充（padding）部分置零，确保向量干净

        return {
            "prompt_embeds": context  # 返回包含文本嵌入的字典
        }


class WanCLIPEncoder(torch.nn.Module):  # 定义 Wan CLIP 视频任务视觉编码器
    """CLIP image encoder for I2V (Image-to-Video) generation."""  # 用于图生视频任务的 CLIP 图像编码器
    def __init__(self, model_name="Wan2.1-T2V-14B"):  # 初始化，默认为 14B 模型版本
        super().__init__()
        self.model_name = model_name  # 保存模型名
        model_path = f"wan_models/{self.model_name}/"  # 设置模型路径
        self.image_encoder = CLIPModel(  # 初始化 CLIP 模型
            dtype=torch.float16,  # 使用半精度以节省内存
            device=torch.device('cpu'),  # 初始加载至 CPU
            checkpoint_path=os.path.join(
                model_path,
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
            tokenizer_path=os.path.join(model_path, "xlm-roberta-large")  # 指定分词器路径
        )
        self.clip = self.image_encoder.model  # 获取底层的 CLIP 模型实例

    @property
    def device(self):  # 设备属性
        # Assume we are always on GPU  # 假设在 GPU 上工作
        return torch.cuda.current_device()

    def forward(self, img):  # 视觉编码前向路径
        """
        Encode image to CLIP features.
        Args:
            img: Image tensor of shape [C, H, W] or [B, C, H, W]
        Returns:
            clip_fea: CLIP features
        """  # 将输入图像编码为 CLIP 特征
        if img.ndim == 3:  # 处理单张图像
            img = img.unsqueeze(1)
        elif img.ndim == 4:  # 处理批次图像
            img = img.transpose(0, 1)
        img = img.to(self.device)  # 移动至 GPU
        clip_encoder_out = self.image_encoder.visual([img])  # 提取视觉特征
        return clip_encoder_out


class WanVAEWrapper(torch.nn.Module):  # 定义 Wan VAE 封装类
    def __init__(self, model_name="Wan2.1-T2V-1.3B"):  # 初始化
        super().__init__()
        self.model_name = model_name  # 保存模型名
        mean = [  # 预定义潜空间的均值，用于反归一化或标准化
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [  # 预定义潜空间的标准差
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)  # 转为张量
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model  # 初始化底层的视频 VAE 模型
        self.model = _video_vae(
            pretrained_path=f"wan_models/{self.model_name}/Wan2.1_VAE.pth",
            z_dim=16,  # 潜空间通道数为 16
        ).eval().requires_grad_(False)  # 设为评估模式且不求导

        # I2V related attributes  # 图生视频（I2V）相关的属性
        self.dtype = torch.bfloat16  # 推理/编码使用的主要数据类型
        self.vae_stride = (4, 8, 8)  # VAE 的下采样步长（时间, 高, 宽）
        self.target_video_length = 81  # 目标视频帧数

    def encode(self, pixel):  # 将像素空间编码为潜空间
        """Batch encode method for I2V."""  # 用于 I2V 的批次编码方法
        device, dtype = pixel[0].device, self.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]  # 计算缩放因子
        output = [
            self.model.encode(u.to(self.dtype).unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]  # 对每一帧执行编码并应用缩放
        return output

    def run_vae_encoder(self, img):  # 运行 VAE 编码器（专门针对训练/控制）
        """
        Encode image for I2V training, returning latent with mask channels.
        Args:
            img: Image tensor of shape [C, H, W]
        Returns:
            List containing vae_encode_out with mask prepended
        """  # 为 I2V 训练编码图像，返回带有掩码通道的潜变量
        img = img.to(torch.bfloat16).cuda()  # 转换类型并移至 GPU
        if img.ndim == 4:
            img = img.squeeze(0)  # 压缩批次维度
        h, w = img.shape[1:]  # 获取图像的高和宽
        lat_h = h // self.vae_stride[1]  # 计算潜空间高度
        lat_w = w // self.vae_stride[2]  # 计算潜空间宽度

        msk = torch.ones(  # 初始化掩码张量，全为 1（表示参考/固定）
            1,
            self.target_video_length,
            lat_h,
            lat_w,
            device=torch.device("cuda"),
        )
        msk[:, 1:] = 0  # 除了第一帧，其余帧掩码设为 0（待生成部分）
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)  # 特殊处理首帧掩码分量
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)  # 重塑维度以匹配模型要求
        msk = msk.transpose(1, 2)[0]  # 转置并选取第一批次
        vae_encode_out = self.encode(  # 执行 VAE 编码
            [
                torch.concat(
                    [
                        torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),  # 插值调整输入图大小
                        torch.zeros(3, self.target_video_length - 1, h, w),  # 补充空帧以对齐长度
                    ],
                    dim=1,
                ).cuda()
            ],
        )[0]
        vae_encode_out = torch.concat([msk, vae_encode_out]).to(torch.bfloat16)  # 拼接掩码和潜变量，转为 bf16
        return [vae_encode_out]

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:  # 将像素转为潜变量的辅助函数
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]  # 归一化参数

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]  # 逐一编码
        output = torch.stack(output, dim=0)  # 堆叠回张量
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]  # 调整轴顺序以匹配下游模型
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:  # 将潜变量解码为像素
        zs = latent.permute(0, 2, 1, 3, 4)  # 换回 VAE 要求的 [C, F, H, W] 布局
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"  # 缓存模式仅支持 Batch Size 为 1

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]  # 反归一化参数

        if use_cache:
            decode_function = self.model.cached_decode  # 使用带缓存的迭代解码
        else:
            decode_function = self.model.decode  # 标准一次性解码

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))  # 解码并裁剪像素范围
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)  # 还原为常规视频张量格式
        return output

    def decode_to_pixel_chunk(self, latent: torch.Tensor, use_cache: bool = False, chunk_size: int = 120) -> torch.Tensor:  # 分块解码以防止 OOM
        """
        Decode latent frames to pixel space.
        
        Args:
            latent: Latent tensor with shape [batch_size, num_frames, num_channels, height, width]
            use_cache: Whether to use cached decoding (for streaming)
            chunk_size: Number of latent frames to decode at once (default 240 to avoid OOM)
        
        Returns:
            Decoded video tensor with shape [batch_size, num_frames, num_channels, height, width]
        """  # 分批次将潜变量帧解码到像素空间。
        # latent shape: [batch_size, num_frames, num_channels, height, width]
        # zs shape after permute: [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            num_frames = u.shape[1]
            if num_frames <= chunk_size:  # 帧数较少，直接完整解码
                decoded = decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                decoded = decoded.cpu()
            else:  # 帧数过多，开启分块解码
                decoded_chunks = []
                for start_idx in range(0, num_frames, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_frames)
                    chunk = u[:, start_idx:end_idx, :, :]  # 提取一块 [C, chunk_frames, H, W]
                    self.model.clear_cache()  # 清理中间缓存
                    decoded_chunk = decode_function(chunk.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                    decoded_chunks.append(decoded_chunk.cpu())  # 转至 CPU 以释放 GPU 显存
                    
                    del decoded_chunk
                    torch.cuda.empty_cache()
                decoded = torch.cat(decoded_chunks, dim=1)  # 拼接所有块
                self.model.clear_cache()
            output.append(decoded)
        
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

class WanDiffusionWrapper(torch.nn.Module):  # 定义 Wan 扩散模型核心封装类
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            use_infinite_attention=False
    ):  # 初始化配置项
        super().__init__()

        if is_causal:  # 如果是因果/自回归模式
            if use_infinite_attention:
                self.model = CausalWanModelInfinity.from_pretrained(  # 加载支持无限长度注意力的模型
                    f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
            else:
                self.model = CausalWanModel.from_pretrained(  # 加载标准因果模型
                    f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")  # 加载常规 T2V 扩散模型
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep  # 非因果扩散中，所有帧共享相同的时间步
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(  # 初始化 Flow-Matching 步进调度器
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)  # 设置默认的 1000 步离散时间步
        # self.seq_len = 1560 * local_attn_size if local_attn_size != -1 else 32760 # [1, 21, 16, 60, 104]
        self.seq_len = 1560 * local_attn_size if local_attn_size > 21 else 32760 # [1, 21, 16, 60, 104]  # 计算注意力机制的最大序列长度
        self.post_init()  # 执行初始化后的一些设置

    def enable_gradient_checkpointing(self) -> None:  # 开启梯度检查点以节省显存
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:  # 添加分类分支（用于判别器等任务）
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!  # 注意：目前该部分硬编码仅适配 1.3B 模型
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),  # 层归一化
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),  # 线性层
            nn.SiLU(),  # 激活函数
            nn.Linear(atten_dim, num_class)  # 最终分类映射
        )
        self._cls_pred_branch.requires_grad_(True)  # 设置为需要梯度
        num_registers = 3  # 定义寄存器 token 数量
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)  # 初始化寄存器 token
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []  # 初始化 GAN 注意力块列表
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)  # 转化为容器以自动管理梯度
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:  # 将预测的流 (Flow) 转换为 x0 (干净数据)
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """  # 使用 Flow-Matching 公式推导，从当前带噪图像和预测的流中还原出 x0 预测值。
        # use higher precision for calculations  # 使用双精度计算以保证数值稳定性
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                         self.scheduler.sigmas,
                                                         self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)  # 寻找最接近的时间步索引
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)  # 获取对应的噪声标准差
        x0_pred = xt - sigma_t * flow_pred  # 执行逆向转换公式
        return x0_pred.to(original_dtype)  # 还原回原始数据类型

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:  # 将 x0 预测转换为流预测 (Flow)
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """  # 逆向公式：已知 x0 预测值，将其转回模型训练所需的流预测格式。
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                       scheduler.sigmas,
                                                       scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t  # 转换公式
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        sink_recache_after_switch=False,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # 主要的前向传播函数，处理各种复杂的生成场景
        prompt_embeds = conditional_dict["prompt_embeds"]  # 获取文本提示词的嵌入向量

        # [B, F] -> [B]  # 处理时间步维度
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]  # 非因果模式下所有帧共用一个时间步
        else:
            input_timestep = timestep  # 因果模式下每个位置可能对应不同时间步

        logits = None
        # X0 prediction  # 执行流预测的核心模型运行部分
        if kv_cache is not None:  # 如果开启了持续生成的增量缓存模式
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),  # 调整维度为 [B, C, F, H, W]
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                sink_recache_after_switch=sink_recache_after_switch,
                clip_fea=clip_fea,
                y=y
            ).permute(0, 2, 1, 3, 4)  # 还原维度顺序
        else:
            if clean_x is not None:  # 教师强制训练模式（Teacher Forcing）
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    sink_recache_after_switch=sink_recache_after_switch,
                    clip_fea=clip_fea,
                    y=y
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:  # 分类模式（用于判别器）
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                        sink_recache_after_switch=sink_recache_after_switch,
                        clip_fea=clip_fea,
                        y=y
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:  # 标准的扩散前向生成
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        sink_recache_after_switch=sink_recache_after_switch,
                        clip_fea=clip_fea,
                        y=y
                    ).permute(0, 2, 1, 3, 4)

        # Convert the predicted flow to x0 (denoised latent)  # 将预测的流转换为 x0（即去噪后的潜变量预测）
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:  # 如果包含分类结果（判别器模式）
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0  # 返回流预测和 x0 预测

    def get_scheduler(self) -> SchedulerInterface:  # 获取并动态绑定调度器方法
        """
        Update the current scheduler with the interface's static method
        """  # 使用接口中的静态方法更新当前调度器的动态绑定，确保方法一致性。
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):  # 对象创建后的自定义初始化步骤
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """  # 在对象实例化后执行的额外初始化操作，目前主要用于绑定调度器方法。
        self.get_scheduler()
