# LongLive Image-to-Video (I2V) 训练指南与实现详解

本文档详细说明如何在 LongLive 项目中运行 I2V 训练，并深入解释了为支持 I2V 模式所做的关键代码修改及其背后的原因。

## 1. 快速开始：训练指令

我们提供了一个一键启动脚本，用于在本地环境（4卡 GPU）启动 I2V 训练。

### 启动命令

```bash
cd /commondocument/group2/LongLive
bash train_i2v_local.sh
```

### 脚本说明 (`train_i2v_local.sh`)

该脚本会自动设置分布式训练环境参数：
- 使用 `torchrun` 启动分布式训练 (nproc_per_node=4)。
- 指定配置文件：`configs/longlive_train_i2v_local.yaml`。
- 设置日志目录：`outputs/longlive_i2v_train`。

```bash
# 核心启动命令示例
torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
    train.py \
    --config configs/longlive_train_i2v_local.yaml \
    --logdir outputs/longlive_i2v_train
```

---

## 2. 关键修改与原因详解

为了将 LongLive 从 T2V 改造为支持 I2V，我们解决了以下几个核心技术障碍。

### 2.1 修复 Pipeline 中的条件传递 (`clip_fea` 和 `y`)

**问题现象**：
训练启动时报错或生成的视频质量极差（无图像参考效果）。调试发现 `dmd.py` 中的 Score Model 和 `pipeline` 中的 Generator 未接收到图像条件。

**原因**：
原版 Self-Forcing Pipeline 仅设计用于 Text-to-Video，代码中并未传递 Image-to-Video 所需的两个关键 Tensor：
1.  `clip_fea`: CLIP 编码的图像全局特征（用于 CrossAttention）。
2.  `y`: VAE 编码的参考帧 Latent（用于与噪声 Concat）。

**修改内容**：
在 `pipeline/self_forcing_training.py` 的 `inference_with_trajectory` 方法（Step 3.3）和 `model/dmd.py` 的 `compute_kl_grad` 方法中，显式提取并传递这两个参数。

```python
# 修改前 (pipeline/self_forcing_training.py)
self.generator(
    noisy_image_or_video=denoised_pred,
    # ... 缺少 y 和 clip_fea
)

# 修改后
self.generator(
    # ...
    clip_fea=conditional_dict.get("clip_fea"),
    # 这里的 y 需要根据当前 Block 进行切片，确保维度匹配
    y=[u[:, current_start:current_end] for u in conditional_dict.get("y")]
)
```

### 2.2 修复 `WanI2VCrossAttention` 签名错误

**问题现象**：
训练报错 `TypeError: WanI2VCrossAttention.forward() got an unexpected keyword argument 'crossattn_cache'`。

**原因**：
Pipeline 为了加速计算，会缓存文本的 Key/Value (`crossattn_cache`)。T2V 模型支持此参数，但 I2V 专用模块 `WanI2VCrossAttention` 的 `forward` 方法签名中遗漏了该参数。

**修改内容**：
更新 `wan/modules/model.py` 中的 `WanI2VCrossAttention` 类，增加 `crossattn_cache` 参数并实现缓存逻辑。

### 2.3 帧数与 Block 大小的对齐 (Frame Alignment)

**问题现象**：
训练报错 `RuntimeError`，提示 Tensor 尺寸不匹配。

**原因**：
LongLive 的 `independent_first_frame` 策略要求：`(总帧数 - 1)` 必须能被 `num_frame_per_block` 整除。
*   **数据集输出**：21 帧 (`[B, 21, C, H, W]`)。
*   **原配置**：`num_frame_per_block: 3`。
*   **冲突**：(21 - 1) = 20，而 20 不能被 3 整除。导致最后一个 Block 尺寸计算错误。

**修改内容**：
修改配置文件 `configs/longlive_train_i2v_local.yaml`：
*   `num_frame_per_block`: 改为 **4**。
*   `num_training_frames`: 设置为 **21**。
*   **验证**：(21 - 1) = 20，20 / 4 = 5 个完整的 Block。逻辑成立。

### 2.4 修正 `WanCLIPEncoder` 维度

**问题现象**：
训练报错 `RuntimeError: Tensors must have same number of dimensions: got 2 and 3`，发生在 `context` 拼接时。

**原因**：
`WanCLIPEncoder` 在输出时多调用了一次 `squeeze(0)`，导致 Batch 维度被意外移除（从 `[B, 257, 1024]` 变为 `[257, 1024]`）。

**修改内容**：
移除 `utils/wan_wrapper.py` 中多余的 `squeeze(0)`，确保保留 Batch 维度。

### 2.5 修复 `train.py` 参数覆盖问题

**问题现象**：
配置文件中设置了 `disable_wandb: true`，但启动训练时仍报错 `Missing key wandb_key`。

**原因**：
`train.py` 使用 `argparse` 的默认值 (`default=False`) 覆盖了配置文件中的设置。即使配置文件显式禁用了 WandB，CLI 参数 (`False`) 会将其重新启用。

**修改内容**：
修改 `train.py`，仅当命令行显式传入 `--disable-wandb` 时才修改配置，否则保留配置文件中的设定。

## 3. 验证结果

经过上述修复，我们在 4 卡环境进行了验证：
*   **稳定性**：训练成功启动并稳定运行超过 1 小时。
*   **Loss 指标**：
    *   `generator_loss`: ~0.3
    *   `critic_loss`: ~0.08
    *   指标处于合理范围，表明模型正在正常学习。
*   **Checkpoint**：成功保存了检查点文件 (`checkpoint_model_*.pt`)。

---
**文档创建时间**: 2026-01-24
