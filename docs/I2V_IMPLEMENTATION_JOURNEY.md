# LongLive I2V 支持：实现历程与关键技术解析

本文档深入回顾 LongLive 项目支持 Image-to-Video (I2V) 流程的开发历程，详细解析核心代码修改、遇到的关键 Bug 及其解决方案，以及背后的设计逻辑。

## 1. I2V 支持的核心修改 (Core Modifications)

为了让原基于 Text-to-Video (T2V) 的 Self-Forcing 框架支持 I2V，我们在以下三个层面进行了核心改造：

### 1.1 数据层 (Data Layer)
*   **输入扩展**: 这里的关键在于不仅仅输入文本 Prompt，还需要输入参考图像。
*   **`ShardingLMDBDataset`**: 修改了 Dataset 类，使其除了返回 text prompt 和 latent 外，额外读取并返回 `img` (参考图像) 字段。
*   **图像编码**: 引入 CLIP Image Encoder，将参考图像编码为 `clip_fea` (Global Context)。
*   **VAE 编码**: 引入 VAE Encoder，将参考图像编码为 Latent `y` (Local Condition)，用于与噪声 Latent 进行拼接 (Concat)。

### 1.2 模型层 (Model Layer)
*   **`WanI2VCrossAttention`**: 这是一个 I2V 专用的注意力模块。原 T2V 模型只关注文本 Context，而 I2V 模型需要同时关注文本和图像特征。
*   **`WanCLIPEncoder`**: 新增的一个编码器模块，用于提取图像的高层语义特征。
*   **`WanDiffusionWrapper`**: 修改了 Forward 签名，使其能够透传 `clip_fea` 和 `y` 到底层的 Transformer Block。

### 1.3 Pipeline 层 (Training Pipeline)
*   **条件注入**: 在 `SelfForcingTrainingPipeline` 的生成循环中，必须显式传递 `clip_fea` 和 `y`。
*   **Block切片逻辑**: 由于 `y` (参考帧 Latent) 是有时序维度的张量，在分块 (Block-wise) 生成时，必须对 `y` 进行正确的切片 (Slicing)，确保每个生成 Block 接收到对应的参考信息。

---

## 2. Bug 猎杀实录 (Bug Hunting & Solutions)

在实现初期，我们遇到了一系列阻碍训练启动的 Bug。以下是详细的复盘：

### 🛑 Bug 1: 训练启动即报错 "Missing key wandb_key"
*   **现象**: `configs/longlive_train_i2v_local.yaml` 中明明设置了 `disable_wandb: true`，但程序依然尝试登录 WandB 并报错。
*   **根因**: `train.py` 使用了 `argparse` 的默认值 (`default=False`)。在合并配置时，CLI 的默认值意外覆盖了 YAML 文件中的 `true` 设置。
*   **解决**: 修改 `train.py` 逻辑，仅当 CLI 显式传入 `--disable-wandb` 时才覆盖配置，否则以 YAML 为准。

### 🛑 Bug 2: 维度不匹配 "RuntimeError: Sizes of tensors must match except in dimension 0"
*   **现象**: 训练崩溃，报错信息指向 Tensor Concat 操作。
*   **根因**: **Frame 与 Block 的数学对齐问题**。
    *   数据集提供 **21 帧**。
    *   启用 `independent_first_frame: true` 后，第1帧作为参考，剩余 **20 帧** 需要生成。
    *   原配置 `num_frame_per_block: 3`。
    *   计算冲突: `20 % 3 = 2` (余 2)，意味着最后一个 Block 长度不足 3 帧，导致模型内部 Tensor 尺寸计算与预期不符。
*   **解决**: 修改配置，将 `num_frame_per_block` 改为 **4**。
    *   `20 % 4 = 0` (整除)，完美适配。

### 🛑 Bug 3: 模型签名错误 "TypeError: unexpected keyword argument 'crossattn_cache'"
*   **现象**: Pipeline 尝试调用模型时报错。
*   **根因**: T2V 训练为了加速，会缓存 Text Encoder 的 Key/Value (`crossattn_cache`)。Pipeline 默认传递此参数，但新集成的 I2V 模块 `WanI2VCrossAttention` 的 `forward` 函数定义中忘记包含此参数。
*   **解决**: 修改 `wan/modules/model.py`，更新 `WanI2VCrossAttention.forward` 签名，加入 `crossattn_cache` 并实现相应的缓存逻辑。

### 🛑 Bug 4: 条件丢失 "AssertionError: clip_fea is not None"
*   **现象**: 训练跑到具体计算 Loss 时报错。
*   **根因**: 数据即虽然加载了图像，但在 Pipeline 的深层调用链中 (特别是 Distillation 的 Generator 循环和 Critic 评分调用)，`clip_fea` 和 `y` 参数在中间环节被丢弃了，没有传下去。
*   **解决**: 全链路打通参数传递。
    *   修改 `trainer/distillation.py`: 在 `fwdbwd_one_step` 中提取 `clip_fea/y`。
    *   修改 `pipeline/self_forcing_training.py`: 在 `inference_with_trajectory` (Step 3.3) 中传递参数。
    *   修改 `model/dmd.py`: 在 `fake_score` 和 `real_score` 调用中传递参数。

### 🛑 Bug 5: 维度错误 "RuntimeError: Tensors must have same number of dimensions"
*   **现象**: 这发生在 `WanCLIPEncoder` 输出特征时。
*   **根因**: `WanCLIPEncoder` 代码中多写了一个 `.squeeze(0)`，导致 Batch Size=1 时 Batch 维度被错误移除 (`[B, L, C]` -> `[L, C]`)。
*   **解决**: 移除多余的 `squeeze` 操作，保持 Batch 维度一致性。

---

## 3. 关键代码解析 (Key Components)

### 3.1 `WanI2VCrossAttention`
这是 I2V 的灵魂组件。不同于普通的 CrossAttention (只看文本)，它拥有两个 Key/Value 源：
1.  **Text Context**: 来自 T5 Encoder 的文本特征。
2.  **Image Context**: 来自 CLIP Encoder 的图像特征 (`clip_fea`)。

代码逻辑中，它会将 Text Embeddings 和 Image Embeddings 在序列长度维度 (Sequence Length) 上进行拼接，然后让 Video Latent 对这个"混合 Context"进行 Attention 操作。

### 3.2 `WanVAEWrapper.run_vae_encoder`
这个函数负责处理参考图像，将其转化为模型可理解的 Condition Latent (`y`)。
*   **Input**: RGB 参考图像 `[B, 3, H, W]`
*   **Process**:
    1.  通过 VAE Encoder 压缩为 Latent。
    2.  **Mask 通道注入**: 它不仅仅返回 Latent，还会在 Channel 维度 Concat 一个 Mask (全1或全0)。这个 Mask 告诉模型："这部分是参考帧，你要强制 Copy" 或者 "后续部分是生成的，你可以自由发挥"。

### 3.3 Pipeline 的 `y` 切片逻辑
在 `pipeline/self_forcing_training.py` 中，我们实现了对 `y` 的切片：

```python
# 伪代码逻辑
current_y = [
    u[:, current_start_frame : current_start_frame + block_size] 
    for u in conditional_dict.get("y")
]
```
这至关重要。因为 `y` 是全视频长度的 Condition (比如 21 帧)，而 Generator 每次只生成一个 Block (比如 4 帧)。我们需要准确地把 `y` 切成小块喂给模型，否则时间维度对不上，模型会混淆"哪帧参考哪帧"。

---
**文档作者**: Antigravity  
**更新日期**: 2026-01-24
