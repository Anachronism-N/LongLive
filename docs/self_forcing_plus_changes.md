# Self-Forcing → Self-Forcing-Plus 代码改动详细记录

> 本文档详细记录了从 `/commondocument/group2/Self-Forcing` 到 `/commondocument/group2/Self-Forcing-Plus` 的所有代码改动，方便后续在其他基于 Self-Forcing 的代码库上进行类似修改。

## 目录

1. [改动概述](#1-改动概述)
2. [新增文件](#2-新增文件)
3. [文件级详细改动](#3-文件级详细改动)
   - [model/base.py](#31-modelbasepy)
   - [trainer/distillation.py](#32-trainerdistillationpy)
   - [utils/wan_wrapper.py](#33-utilswan_wrapperpy)
   - [utils/dataset.py](#34-utilsdatasetpy)
   - [pipeline/self_forcing_training.py](#35-pipelineself_forcing_trainingpy)
   - [pipeline/causal_inference.py](#36-pipelinecausal_inferencepy)
   - [pipeline/__init__.py](#37-pipeline__init__py)
   - [model/dmd.py](#38-modeldmdpy)
   - [configs](#39-configs-配置文件)
4. [迁移指南](#4-迁移指南)

---

## 1. 改动概述

Self-Forcing-Plus 相比 Self-Forcing 的主要改动包括：

| 改动类别 | 具体内容 |
|---------|---------|
| **14B 模型支持** | 新增对 `Wan2.1-T2V-14B` 模型的全面支持，包括动态 `num_transformer_blocks` (1.3B: 30, 14B: 40) |
| **生成器类型配置** | 新增 `generator_type` 配置项，支持 `causal` (因果) 和 `bidirectional` (双向) 两种生成器 |
| **I2V 图像编码** | 新增 `WanCLIPEncoder` 图像编码器支持 Image-to-Video 训练 |
| **双向训练 Pipeline** | 新增 `BidirectionalTrainingPipeline` 用于双向生成器训练 |
| **数据集扩展** | 新增 `TextFolderDataset` 支持从文件夹读取文本提示 |
| **Checkpoint 恢复** | 改进训练恢复逻辑，支持分别加载 generator、critic 和 EMA |
| **KV Cache 动态化** | 将硬编码的 12 attention heads 改为动态 `num_transformer_blocks` |

---

## 2. 新增文件

### 2.1 pipeline/bidirectional_training.py (全新文件)

```python
# 文件路径: pipeline/bidirectional_training.py
# 大小: 4416 bytes
# 用途: 双向生成器的训练 pipeline

from typing import List
import torch
from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
import torch.distributed as dist


class BidirectionalTrainingPipeline(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
    ):
        super().__init__()
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

    def generate_and_sync_list(self, num_denoising_steps, device):
        # ... 同步随机索引，与 SelfForcingTrainingPipeline 类似但只生成1个索引

    def inference_with_trajectory(self, noise: torch.Tensor, clip_fea, y, **conditional_dict) -> torch.Tensor:
        # 关键区别：
        # 1. 不使用 KV cache
        # 2. 接受 clip_fea 和 y 参数用于 I2V
        # 3. 对所有帧使用相同的 timestep (uniform_timestep)
```

### 2.2 configs/self_forcing_14b_dmd.yaml

新增 14B 模型专用配置，关键新增字段：
- `generator_type: bidirectional`
- `generator_name: Wan2.1-T2V-14B`
- `data_type: text_folder`
- `data_max_count: 30000`

### 2.3 configs/self_forcing_14b_i2v_dmd.yaml

新增 14B 模型 I2V 训练配置。

---

## 3. 文件级详细改动

### 3.1 model/base.py

#### 3.1.1 新增导入

```diff
- from pipeline import SelfForcingTrainingPipeline
+ from pipeline import SelfForcingTrainingPipeline, BidirectionalTrainingPipeline

- from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
+ from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, WanCLIPEncoder
```

#### 3.1.2 BaseModel.__init__ 新增

```diff
  class BaseModel(nn.Module):
      def __init__(self, args, device):
          super().__init__()
+         self.is_causal = args.generator_type == "causal"
+         self.i2v = args.i2v
          self._initialize_models(args, device)
```

#### 3.1.3 _initialize_models 改动

```diff
  def _initialize_models(self, args, device):
-     self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
-     self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
+     self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-14B")
+     self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-14B")
+     self.generator_name = getattr(args, "generator_name", "Wan2.1-T2V-14B")

-     self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
+     self.generator = WanDiffusionWrapper(
+         **getattr(args, "model_kwargs", {}),
+         model_name=self.generator_name,
+         is_causal=self.is_causal
+     )

-     self.text_encoder = WanTextEncoder()
+     self.text_encoder = WanTextEncoder(model_name=self.generator_name)

-     self.vae = WanVAEWrapper()
+     self.vae = WanVAEWrapper(model_name=self.generator_name)

+     # 新增：I2V 需要图像编码器
+     if self.i2v:
+         self.image_encoder = WanCLIPEncoder(model_name=self.generator_name)
+         self.image_encoder.requires_grad_(False)
```

#### 3.1.4 SelfForcingModel._run_generator 新增参数

```diff
  def _run_generator(
      self,
      image_or_video_shape,
      conditional_dict: dict,
-     initial_latent: torch.tensor = None
+     initial_latent: torch.tensor = None,
+     clip_fea: torch.Tensor = None,
+     y: torch.Tensor = None
  ) -> Tuple[torch.Tensor, torch.Tensor]:
```

调用时传递新参数：
```diff
  pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
      noise=torch.randn(noise_shape, device=self.device, dtype=self.dtype),
+     clip_fea=clip_fea,
+     y=y,
      **conditional_dict,
  )
```

#### 3.1.5 _consistency_backward_simulation 新增参数

```diff
  def _consistency_backward_simulation(
      self,
      noise: torch.Tensor,
+     clip_fea: torch.Tensor,
+     y: torch.Tensor,
      **conditional_dict: dict
  ) -> torch.Tensor:
```

#### 3.1.6 _initialize_inference_pipeline 改为分支逻辑

```diff
  def _initialize_inference_pipeline(self):
-     self.inference_pipeline = SelfForcingTrainingPipeline(
-         denoising_step_list=self.denoising_step_list,
-         scheduler=self.scheduler,
-         generator=self.generator,
-         num_frame_per_block=self.num_frame_per_block,
-         independent_first_frame=self.args.independent_first_frame,
-         same_step_across_blocks=self.args.same_step_across_blocks,
-         last_step_only=self.args.last_step_only,
-         num_max_frames=self.num_training_frames,
-         context_noise=self.args.context_noise
-     )
+     if self.is_causal:
+         self.inference_pipeline = SelfForcingTrainingPipeline(
+             model_name=self.generator_name,
+             denoising_step_list=self.denoising_step_list,
+             scheduler=self.scheduler,
+             generator=self.generator,
+             num_frame_per_block=self.num_frame_per_block,
+             independent_first_frame=self.args.independent_first_frame,
+             same_step_across_blocks=self.args.same_step_across_blocks,
+             last_step_only=self.args.last_step_only,
+             num_max_frames=self.num_training_frames,
+             context_noise=self.args.context_noise
+         )
+     else:
+         self.inference_pipeline = BidirectionalTrainingPipeline(
+             model_name=self.generator_name,
+             denoising_step_list=self.denoising_step_list,
+             scheduler=self.scheduler,
+             generator=self.generator,
+         )
```

---

### 3.2 trainer/distillation.py

#### 3.2.1 新增导入

```diff
- from utils.dataset import ShardingLMDBDataset, cycle
- from utils.dataset import TextDataset
+ from utils.dataset import ShardingLMDBDataset, cycle
+ from utils.dataset import TextDataset, TextFolderDataset
```

#### 3.2.2 新增 image_encoder FSDP 封装

```diff
  # 在 fake_score FSDP 封装后新增
+ if self.config.i2v:
+     self.model.image_encoder = fsdp_wrap(
+         self.model.image_encoder,
+         sharding_strategy=config.sharding_strategy,
+         mixed_precision=config.mixed_precision,
+         wrap_strategy=config.image_encoder_fsdp_wrap_strategy,
+         min_num_params=int(5e6),
+         cpu_offload=getattr(config, "image_encoder_cpu_offload", False)
+     )
+     self.model.vae = self.model.vae.to(
+         device=self.device, dtype=torch.bfloat16)
+
+ elif not config.no_visualize or config.load_raw_video:
+     self.model.vae = self.model.vae.to(
+         device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)
```

#### 3.2.3 数据集初始化改动

```diff
  # Step 3: Initialize the dataloader
- if self.config.i2v or os.path.isdir(config.data_path):
-     dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
- else:
-     dataset = TextDataset(config.data_path)
+ if self.config.i2v:
+     dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
+ else:
+     if self.config.data_type == "text_folder":
+         data_max_count = config.get("data_max_count", 30000)
+         dataset = TextFolderDataset(config.data_path, data_max_count)
+     elif self.config.data_type == "text_file":
+         dataset = TextDataset(config.data_path)
+     else:
+         raise ValueError("Invalid data type")
```

#### 3.2.4 EMA 初始化改动

```diff
- ema_weight = config.ema_weight
- self.generator_ema = None
- if (ema_weight is not None) and (ema_weight > 0.0):
-     print(f"Setting up EMA with weight {ema_weight}")
-     self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)
+ self.ema_weight = config.get("ema_weight", -1.0)
+ self.ema_start_step = config.get("ema_start_step", 0)
+ self.generator_ema = None
+ if (self.ema_weight > 0.0) and (self.step >= self.ema_start_step):
+     print(f"Setting up EMA with weight {self.ema_weight}")
+     self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)
```

#### 3.2.5 Checkpoint 恢复逻辑改进

```diff
- if getattr(config, "generator_ckpt", False):
-     print(f"Loading pretrained generator from {config.generator_ckpt}")
-     state_dict = torch.load(config.generator_ckpt, map_location="cpu")
-     if "generator" in state_dict:
-         state_dict = state_dict["generator"]
-     elif "model" in state_dict:
-         state_dict = state_dict["model"]
-     self.model.generator.load_state_dict(
-         state_dict, strict=True
-     )
+ if getattr(config, "resume_ckpt", False):
+     print(f"Resuming training from {config.resume_ckpt}")
+     
+     # Set resume step
+     if getattr(config, "resume_step", False):
+         self.step = config.resume_step
+         print(f"Resuming from step {self.step}")
+
+     # Load generator_ema checkpoint (if exists)
+     generator_ema_path = os.path.join(config.resume_ckpt, "generator_ema.pt")
+     if os.path.exists(generator_ema_path):
+         if self.generator_ema is None and self.ema_weight > 0.0:
+             print("Initializing EMA for resume...")
+             generator_state_dict = torch.load(generator_ema_path, map_location="cpu")
+             self.model.generator.load_state_dict(generator_state_dict, strict=True)
+             self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)
+             print("Generator EMA checkpoint loaded successfully")
+     
+     # Load generator checkpoint
+     generator_path = os.path.join(config.resume_ckpt, "generator.pt")
+     if os.path.exists(generator_path):
+         generator_state_dict = torch.load(generator_path, map_location="cpu")
+         self.model.generator.load_state_dict(generator_state_dict, strict=True)
+     
+     # Load critic checkpoint
+     critic_path = os.path.join(config.resume_ckpt, "critic.pt")
+     if os.path.exists(critic_path):
+         critic_state_dict = torch.load(critic_path, map_location="cpu")
+         self.model.fake_score.load_state_dict(critic_state_dict, strict=True)
```

#### 3.2.6 fwdbwd_one_step 新增 clip_fea 和 y

```diff
  # Step 2 后新增：
+ if self.config.i2v:
+     img = batch["img"].to(self.device).squeeze(0)
+     clip_fea = self.model.image_encoder(img)
+     y = self.model.vae.run_vae_encoder(img)
+ else:
+     clip_fea = None
+     y = None

  # generator_loss 调用新增参数：
  generator_loss, generator_log_dict = self.model.generator_loss(
      image_or_video_shape=image_or_video_shape,
      conditional_dict=conditional_dict,
      unconditional_dict=unconditional_dict,
      clean_latent=clean_latent,
      initial_latent=image_latent if self.config.i2v else None,
+     clip_fea=clip_fea,
+     y=y
  )

+ torch.cuda.empty_cache()  # 新增显存清理

  # critic_loss 调用同样新增 clip_fea 和 y 参数
```

#### 3.2.7 train 循环新增日志

```diff
  while True:
+     if self.is_main_process:
+         print(f"training step {self.step} ...")
      TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
```

---

### 3.3 utils/wan_wrapper.py

#### 3.3.1 新增导入

```diff
+ import os
+ from wan.modules.clip import CLIPModel
```

#### 3.3.2 WanTextEncoder 支持动态模型名

```diff
  class WanTextEncoder(torch.nn.Module):
-     def __init__(self) -> None:
+     def __init__(self, model_name="Wan2.1-T2V-14B") -> None:
          super().__init__()
+         self.model_name = model_name

          self.text_encoder.load_state_dict(
-             torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", ...)
+             torch.load(f"wan_models/{self.model_name}/models_t5_umt5-xxl-enc-bf16.pth", ...)
          )
          
          self.tokenizer = HuggingfaceTokenizer(
-             name="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/", ...)
+             name=f"wan_models/{self.model_name}/google/umt5-xxl/", ...)
```

#### 3.3.3 新增 WanCLIPEncoder 类 (全新)

```python
class WanCLIPEncoder(torch.nn.Module):
    def __init__(self, model_name="Wan2.1-T2V-14B"):
        super().__init__()
        self.model_name = model_name
        self.image_encoder = CLIPModel(
            dtype=torch.float16,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(
                f"wan_models/{self.model_name}/",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            )
        )

    @property
    def device(self):
        return torch.cuda.current_device()

    def forward(self, img):
        img = img[:, None, :, :].to(self.device)
        clip_encoder_out = self.image_encoder.visual([img]).squeeze(0)
        return clip_encoder_out
```

#### 3.3.4 WanVAEWrapper 支持动态模型名并新增 run_vae_encoder

```diff
  class WanVAEWrapper(torch.nn.Module):
-     def __init__(self):
+     def __init__(self, model_name="Wan2.1-T2V-14B"):
          super().__init__()
+         self.model_name = model_name
          
          self.model = _video_vae(
-             pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
+             pretrained_path=f"wan_models/{self.model_name}/Wan2.1_VAE.pth",
              z_dim=16,
          )

+         self.dtype = torch.bfloat16
+         self.vae_stride = (4, 8, 8)
+         self.target_video_length = 81

+     # 新增方法
+     def encode(self, pixel):
+         """批量编码方法"""
+         ...

+     # 新增方法 - 用于 I2V 训练
+     def run_vae_encoder(self, img):
+         """
+         为 I2V 训练编码图像，返回包含 mask 的 latent
+         输出格式: [mask_channels, vae_latent_channels]
+         """
+         img = img.to(torch.bfloat16).cuda()
+         h, w = img.shape[1:]
+         lat_h = h // self.vae_stride[1]
+         lat_w = w // self.vae_stride[2]
+
+         msk = torch.ones(1, self.target_video_length, lat_h, lat_w, device="cuda")
+         msk[:, 1:] = 0
+         msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
+         msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
+         msk = msk.transpose(1, 2)[0]
+         
+         vae_encode_out = self.encode([...])
+         vae_encode_out = torch.concat([msk, vae_encode_out]).to(torch.bfloat16)
+         return [vae_encode_out]
```

#### 3.3.5 WanDiffusionWrapper 改动

```diff
  class WanDiffusionWrapper(torch.nn.Module):
      def __init__(
              self,
-             model_name="Wan2.1-T2V-1.3B",
+             model_name="Wan2.1-T2V-14B",
              ...
      ):
          super().__init__()
+         self.model_name = model_name
+         self.dim = 5120 if "14B" in model_name else 1536

      # adding_cls_branch 中的维度改动
-     nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
+     nn.Linear(atten_dim * 3 + time_embed_dim, self.dim),

      # forward 新增参数
      def forward(
          self,
          noisy_image_or_video: torch.Tensor, conditional_dict: dict,
          timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
          ...
+         clip_fea: Optional[torch.Tensor] = None,
+         y: Optional[torch.Tensor] = None
      ) -> torch.Tensor:
          
          # 所有 model 调用都需要传递 clip_fea 和 y
          flow_pred = self.model(
              ...,
+             clip_fea=clip_fea,
+             y=y
          )
```

---

### 3.4 utils/dataset.py

#### 3.4.1 新增 TextFolderDataset 类

```python
class TextFolderDataset(Dataset):
    """从文件夹中读取 .txt 文件作为文本提示"""
    def __init__(self, data_path, max_count=30000):
        self.texts = []
        count = 1
        for file in os.listdir(data_path):
            if file.endswith(".txt"):
                with open(os.path.join(data_path, file), "r") as f:
                    text = f.read().strip()
                    self.texts.append(text)
                    count += 1
                    if count > max_count:
                        break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"prompts": self.texts[idx], "idx": idx}
```

#### 3.4.2 ShardingLMDBDataset 新增 img 字段

```diff
  def __getitem__(self, idx):
      ...
+     img = retrieve_row_from_lmdb(
+         self.envs[shard_id],
+         "img", np.uint8, local_idx,
+         shape=(480, 832, 3)
+     )
+     img = Image.fromarray(img)
+     img = TF.to_tensor(img).sub_(0.5).div_(0.5)

      return {
          "prompts": prompts,
          "ode_latent": torch.tensor(latents, dtype=torch.float32),
+         "img": img
      }
```

需要新增导入：
```diff
+ import torchvision.transforms.functional as TF
```

---

### 3.5 pipeline/self_forcing_training.py

#### 3.5.1 构造函数改动

```diff
  class SelfForcingTrainingPipeline:
      def __init__(self,
+                  model_name: str,
                   denoising_step_list: List[int],
                   ...):
          super().__init__()
+         self.model_name = model_name

          # 动态设置 transformer blocks 数量
-         self.num_transformer_blocks = 30
+         self.num_transformer_blocks = 40 if "14B" in model_name else 30

+         self.crossattn_cache = None  # 初始化为 None
```

#### 3.5.2 inference_with_trajectory 新增参数

```diff
  def inference_with_trajectory(
          self,
          noise: torch.Tensor,
+         clip_fea: Optional[torch.Tensor] = None,
+         y: Optional[torch.Tensor] = None,
          initial_latent: Optional[torch.Tensor] = None,
          return_sim_step: bool = False,
          **conditional_dict
  ) -> torch.Tensor:
```

#### 3.5.3 KV cache 动态 heads

```diff
  def _initialize_kv_cache(self, batch_size, dtype, device):
      for _ in range(self.num_transformer_blocks):
          kv_cache1.append({
-             "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
-             "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
+             "k": torch.zeros([batch_size, self.kv_cache_size, self.num_transformer_blocks, 128], dtype=dtype, device=device),
+             "v": torch.zeros([batch_size, self.kv_cache_size, self.num_transformer_blocks, 128], dtype=dtype, device=device),
              ...
          })

  def _initialize_crossattn_cache(self, batch_size, dtype, device):
      for _ in range(self.num_transformer_blocks):
          crossattn_cache.append({
-             "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
-             "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
+             "k": torch.zeros([batch_size, 512, self.num_transformer_blocks, 128], dtype=dtype, device=device),
+             "v": torch.zeros([batch_size, 512, self.num_transformer_blocks, 128], dtype=dtype, device=device),
              ...
          })
```

> [!WARNING]
> **KV cache heads 数量的改动可能需要进一步确认！** 原代码使用 `12` 作为 attention heads 数量，新代码使用 `self.num_transformer_blocks`（30 或 40），这两个值含义不同。建议核实实际的 attention heads 数量配置。

---

### 3.6 pipeline/causal_inference.py

文件大小从 14554 bytes 变为 14154 bytes，减少约 400 bytes。具体改动需进一步分析。

---

### 3.7 pipeline/__init__.py

```diff
- # 原导出
+ from pipeline.bidirectional_training import BidirectionalTrainingPipeline
```

---

### 3.8 model/dmd.py

文件大小从 15484 bytes 增长到 16281 bytes，新增约 800 bytes。主要改动包括传递 `clip_fea` 和 `y` 参数到 `_run_generator` 和相关函数调用。

---

### 3.9 configs 配置文件

#### 新增配置项

| 配置项 | 类型 | 说明 | 示例值 |
|-------|------|------|--------|
| `generator_type` | string | 生成器类型 | `"causal"` 或 `"bidirectional"` |
| `generator_name` | string | 生成器模型名称 | `"Wan2.1-T2V-14B"` |
| `data_type` | string | 数据类型 | `"text_folder"` 或 `"text_file"` |
| `data_max_count` | int | TextFolderDataset 最大数量 | `30000` |
| `resume_ckpt` | string | 恢复训练的 checkpoint 路径 | `"checkpoints/step_1000"` |
| `resume_step` | int | 恢复训练的步数 | `1000` |
| `image_encoder_fsdp_wrap_strategy` | string | 图像编码器 FSDP 策略 | `"size"` |
| `image_encoder_cpu_offload` | bool | 图像编码器 CPU offload | `false` |

#### 配置示例对比

**原版 (self_forcing_dmd.yaml):**
```yaml
real_name: Wan2.1-T2V-14B
generator_ckpt: checkpoints/ode_init.pt
data_path: prompts/vidprom_filtered_extended.txt
sharding_strategy: hybrid_full
```

**新版 (self_forcing_14b_dmd.yaml):**
```yaml
real_name: Wan2.1-T2V-14B
fake_name: Wan2.1-T2V-14B
generator_type: bidirectional
generator_name: Wan2.1-T2V-14B
data_type: text_folder
data_path: prompts/good_prompts/
data_max_count: 30000
sharding_strategy: full
# 删除了 generator_ckpt，改用 resume_ckpt
```

---

## 4. 迁移指南

如需在其他基于 Self-Forcing 的代码上应用类似改动，请按以下顺序操作：

### Step 1: 更新 utils/wan_wrapper.py

1. 添加 `os` 和 `CLIPModel` 导入
2. 修改 `WanTextEncoder`、`WanVAEWrapper` 构造函数接受 `model_name` 参数
3. 新增 `WanCLIPEncoder` 类
4. 在 `WanVAEWrapper` 中新增 `run_vae_encoder` 方法
5. 修改 `WanDiffusionWrapper`:
   - 添加 `self.dim` 动态计算
   - `forward` 方法新增 `clip_fea` 和 `y` 参数

### Step 2: 更新 utils/dataset.py

1. 新增 `TextFolderDataset` 类
2. 修改 `ShardingLMDBDataset.__getitem__` 返回 `img` 字段
3. 添加 `torchvision.transforms.functional` 导入

### Step 3: 新增 pipeline/bidirectional_training.py

创建 `BidirectionalTrainingPipeline` 类。

### Step 4: 更新 pipeline/self_forcing_training.py

1. 构造函数新增 `model_name` 参数
2. 动态计算 `num_transformer_blocks`
3. `inference_with_trajectory` 新增 `clip_fea` 和 `y` 参数
4. KV cache 初始化使用动态 heads 数量

### Step 5: 更新 pipeline/__init__.py

导出 `BidirectionalTrainingPipeline`。

### Step 6: 更新 model/base.py

1. 新增 `is_causal` 和 `i2v` 属性
2. 导入 `BidirectionalTrainingPipeline` 和 `WanCLIPEncoder`
3. 修改 `_initialize_models` 支持动态模型名和 image_encoder
4. 修改 `_run_generator` 和 `_consistency_backward_simulation` 接受 `clip_fea` 和 `y`
5. 修改 `_initialize_inference_pipeline` 支持双向 pipeline

### Step 7: 更新 model/dmd.py

传递 `clip_fea` 和 `y` 参数。

### Step 8: 更新 trainer/distillation.py

1. 导入 `TextFolderDataset`
2. 新增 image_encoder FSDP 封装
3. 修改数据集初始化逻辑
4. 改进 EMA 初始化和 checkpoint 恢复
5. `fwdbwd_one_step` 中处理 `clip_fea` 和 `y`

### Step 9: 更新配置文件

新增必要的配置项。

---

> 文档生成时间: 2026-01-24
> 对比版本: Self-Forcing (原版) vs Self-Forcing-Plus (改进版)
