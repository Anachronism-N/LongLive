# LongLive T2V 训练流程详细解析

本文档梳理了 LongLive T2V（Text-to-Video）训练的核心流程，从入口脚本开始，追踪代码在各个文件和函数间的跳转，并解释关键代码块的含义。

LongLive 基于 **DMD (Distribution Matching Distillation)** 算法进行训练，旨在让一个学生模型（Generator）通过匹配真实数据分布（由 Target Teacher 提供）和自身的伪分布（由 Fake Critic 评估），来实现高效的少步数甚至单步生成。

---

## 1. 训练入口：`train.py`

一切从 `train.py` 启动。

- **L41-43**:
  ```python
  if config.trainer == "score_distillation":
      trainer = ScoreDistillationTrainer(config)
  trainer.train()
  ```
  **含义**：解析配置文件，初始化 `ScoreDistillationTrainer`（位于 `trainer/distillation.py`），并调用其 `train()` 方法启动主训练循环。

---

## 2. 主训练循环：`trainer/distillation.py`

`ScoreDistillationTrainer`（继承自 `Trainer`）控制着整个模型的优化生命周期。

- **`train()` 方法 (约 L1180)**：
  该方法包含了一个 `while` 循环，遍历数据加载器直至达到最大迭代次数。
  - **L1210-L1320**: 梯度累积循环。分别调用 `self.fwdbwd_one_step(batch, True)` 训练 Generator，然后调用 `self.fwdbwd_one_step(batch, False)` 训练 Critic。对于长视频由于显存问题，可能会调用 `fwdbwd_one_step_streaming` 进行流式的梯度累加。

- **`fwdbwd_one_step()` 方法 (L833)**：
  处理单步前向与反向传播的业务逻辑：
  - **L848**: 获取文本条件的 embedding (`self.model.text_encoder`)。
  - **L867**: 调用 `self.model.generator_loss(...)` —— 计算生成器（Generator）的损失并反向传播。
  - **L888**: 调用 `self.model.critic_loss(...)` —— 计算判别器（Critic）的损失并反向传播。

*(注：这里的 `self.model` 具体是 `DMD` 类的实例，定义在 `model/dmd.py` 中。)*

---

## 3. DMD 损失计算：`model/dmd.py`

该文件是 DMD 蒸馏算法的核心数学实现。`DMD` 类继承自 `model/base.py` 中的 `SelfForcingModel`。

### 3.1 训练生成器 (`generator_loss`, L212)
其目标是使生成器产生的视频分布向教师模型的真实分布靠拢。
- **L241 -> `self._run_generator(...)`**：
  跳转至基类 `model/base.py`。这是通过给定的提示词，从纯噪声开始“反向模拟”（backward simulation）出一段 Fake 视频（`pred_image`）。这代替了传统 GAN 需要外部真实数据集的做法。
- **L258 -> `self.compute_distribution_matching_loss(...)`**：
  计算分布匹配损失。这涉及复杂的 KL 散度梯度估计。
  - **跳转至 L144 (`compute_distribution_matching_loss`)**:
    - **L188**: 为刚生成的 Fake 视频添加纯随机噪声，模拟扩散过程的某个中间 timestep。
    - **L196 -> `self._compute_kl_grad(...)`**:
      - **L83**: 传入 Fake Critic (自身分布评估模型) 预测噪声（`pred_fake_image`）。
      - **L108**: 传入 Target Teacher（预训练的 Wan 原版模型）预测噪声（`pred_real_image`）。
      - **L129**: 计算 DMD 伪梯度：`grad = pred_fake_image - pred_real_image`。这就是让假分布逼近真分布的优化方向。
    - **L208**: 最终损失函数被定义为回归损失：`0.5 * MSE(original_latent, original_latent - grad)`，强制模型吸收这个差值梯度。

### 3.2 训练判别器/批评器 (`critic_loss`, L282)
其目标是让 Fake Critic 准确认识生成器目前的“错误”分布，以便给 Generator 提供正确的指导。
- **L312 -> `self._run_generator(...)`**：
  同样，先自回归生成一段当前的 Fake 视频（`generated_image`）。
- **L345**: `self.scheduler.add_noise(...)`
  为 Fake 视频加上随机时间步（`critic_timestep`）的噪声。
- **L351**:
  将加噪后的视频传入 `self.fake_score`（即 Fake Critic），得到它的去噪预测（`pred_fake_image`）。
- **L377**:
  计算标准的去噪损失（Denoising Loss, 即 `self.denoising_loss_func`）。通过让 Fake Critic 用标准的扩散机制去去噪 Generator 产生的数据，Critic 学习并跟踪到了当前 Generator 的“伪分布（Fake Distribution）”。

---

## 4. 视频生成展开：`model/base.py` && `pipeline/self_forcing_training.py`

在每次计算 Generator 和 Critic 的 Loss 之前，都需要调用 `_run_generator` 无需梯度地生成视频，这就是典型的自回归扩散模拟流程。

- **`model/base.py` -> `_run_generator()` (L144)**：
  计算需要生成的帧数和 Block 数，初始化相应的噪声 `noise_shape`。
  - **L167 -> `self._consistency_backward_simulation(...)`**：
    这是反向模拟的具体入口。
  - **L229 -> `self.inference_pipeline.inference_with_trajectory(...)`**：
    代码实际跳转到 `pipeline/self_forcing_training.py`。

- **`pipeline/self_forcing_training.py` -> `inference_with_trajectory()`**：
  这个管道是对 `wan/modules/causal_model.py` 的高级封装，实现了 **Streaming / Block-wise（流式/分块）长视频生成**。
  - **步骤 1**: 初始化全局或局部的 KV Cache。
  - **步骤 2: Temporal denoising loop (时间去噪循环)**。它将长视频切分为多个包含 `num_frame_per_block` 帧的数据块，在时间轴上滑动窗口。
  - **步骤 2.1: Spatial denoising loop (空间去噪循环)**。在每个 Block 内部，按照 `denoising_step_list`（如 [1000, 750, 500, 250]），调用 Transformer 模型进行多步去噪，生成一个块的 Latent。
  - 生成出来的 Block 和历史记录拼接，并存储在 KV Cache 中由下一帧读取使用，最终得到一条拥有上下文依赖关系的长视频。
