# LongLive 架构与训练策略解析

本文档主要阐述 LongLive 项目相对于基础版 Self-Forcing 的架构演进，以及当前的训练策略。

## 1. LongLive 在 Self-Forcing 基础上的修改 (Self-Forcing vs LongLive)

LongLive 项目是基于 Self-Forcing 框架深度定制的 Ultra 版本，主要在以下维度进行了架构升级：

### 1.1 模型规模支持 (14B Scaling)
*   **原版 (Self-Forcing)**: 主要针对 1.3B 参数量级模型设计。
*   **LongLive**: 
    *   全面适配 **14B (Wan2.1-T2V-14B)** 模型。
    *   实现了动态 Transformer Layer 配置 (30层 vs 40层)。
    *   优化了 FSDP (Fully Sharded Data Parallel) 策略以适应大模型显存需求。

### 1.2 生成机制扩展 (Generator Types)
*   **原版**: 仅支持标准的 Causal (自回归/因果) 生成模式。
*   **LongLive**: 
    *   引入了 **Bidirectional (双向)** 生成支持。
    *   在 `pipeline` 和 `inference` 层支持非因果的 Context 用于更复杂的视频生成任务。

### 1.3 I2V (Image-to-Video) 深度集成
这是 LongLive 最显著的改动点。原版 Self-Forcing 主要是 Text-to-Video 框架。LongLive 通过以下模块实现了原生 I2V 支持：
*   **WanCLIPEncoder**: 引入视觉编码器。
*   **WanI2VCrossAttention**: 改造注意力机制，支持 Vision-Text Dual Context。
*   **Condition Injection**: 在 Pipeline 全流程打通了 Reference Latent (`y`) 的注入通道。

---

## 2. 训练策略演进 (Training Strategy)

### 2.1 原始 LongLive (Two-Stage) 策略
早期的 LongLive 或标准视频生成训练通常遵循两阶段范式：

1.  **Stage 1: SFT (Supervised Fine-Tuning)**
    *   目标：让模型"学会"视频的基本分布。
    *   方法：使用高质量视频数据，进行标准的 Next Token Prediction (如果是 Autoregressive) 或 Noise Prediction (如果是 Diffusion) 训练。
    *   特点：数据驱动，不涉及复杂的 Loss 设计。

2.  **Stage 2: Distillation / Alignment**
    *   目标：提升生成质量，加速推理。
    *   方法：使用特定的 Distillation Loss (如 Video Distillation)。

### 2.2 当前 LongLive (One-Stage Score Distillation) 策略

我们现在采用的是一种更为端到端的 **Score Distillation (分数蒸馏)** 策略，这在 `trainer/distillation.py` 中体现得尤为明显。

#### 核心机制：DMD (Distribution Matching Distillation)
当前的训练不再严格区分 SFT 和 Distillation 阶段，而是直接进行**分布匹配蒸馏**。

*   **Teacher-Student 架构**:
    *   **Teacher (Fake Score / Real Score)**: 冻结的 (或部分冻结的) 预训练模型，作为评分器 (Critic/Score Model)。
    *   **Student (Generator)**: 需要训练的生成模型。

*   **Self-Forcing Loop**:
    1.  **Generation**: Generator 从噪声生成视频 Latent (Block by Block)。为了节省显存和通过长视频训练，利用了 **KV Cache** 传递历史信息 (Self-Forcing 机制的核心)。
    2.  **Scoring**: 将生成的视频 Latent 喂给 Score Model (Critic)。
    3.  **Optimization**: 
        *   **Critic Loss**: 训练 Critic 更好地区分"真实视频分布"和"生成视频分布"。
        *   **Generator Loss**: 利用 Critic 给出的梯度 (Score Gradient) 更新 Generator，使其生成的分布向真实分布靠拢。

#### 这种策略的优势
*   **数据效率**: 通过 KV Cache 和分块生成，可以在有限显存下训练超长视频 (Long Context)。
*   **一步到位**: 直接优化生成质量，跳过了繁琐的 SFT -> RLHF/Distillation 流程。
*   **I2V 适配**: 这种 Generator-Critic 结构非常适合 I2V，因为 Reference Image 提供了强有力的 Conditional Guidance，加速了分布匹配的收敛。

### 总结
我们目前的训练脚本 (`train_i2v_local.sh`) 运行的就是这个 **Score Distillation** 过程。它加载预训练的 Teacher 权重，直接对 Generator 进行微调，使其能够根据 Image Condition 生成连贯的长视频。

---
**文档作者**: Antigravity
**更新日期**: 2026-01-24
