# LongLive I2V 开发与调试对话记录

*本文档为 AI 助手与开发者的对话历史备份，记录了 LongLive 项目从 T2V 向 I2V 改造过程中的核心调试步骤与技术讨论。*

---

## 阶段一：I2V 训练维度的初步排查

**开发者**: （发起了关于 I2V 改造的任务，此时训练在 Pipeline 阶段出错）
**AI**: 观察到训练由于 `WanCLIPEncoder` & `run_vae_encoder` 触发 `RuntimeError: Tensors must have same number of dimensions: got 2 and 3`。错误发生在 `causal_model.py` 中拼接输入 `x` 和条件 `y` 的时候。
> 分析发现是 `WanCLIPEncoder` 的输出多了一次 `squeeze(0)`，导致丢失了 Batch 维度。
**解决方案**: 移除了 `utils/wan_wrapper.py` 中 `clip_encoder_out = self.image_encoder.visual([img]).squeeze(0)` 的 `.squeeze(0)`，保留 Batch 维度。

---

## 阶段二：解决 crossattn_cache 签名错误

**开发者**: （执行训练脚本测试）
**报错**: `TypeError: WanI2VCrossAttention.forward() got an unexpected keyword argument 'crossattn_cache'`
**AI**: 检查发现 `WanI2VCrossAttention` 的 `forward` 方法缺失了 `crossattn_cache` 参数，而 Pipeline 默认会传递此参数以加速文本特征的缓存。
**解决方案**: 
1. 在 `wan/modules/model.py` 中修改 `WanI2VCrossAttention.forward`，增加 `crossattn_cache=None` 并实现了 KV 缓存逻辑。
2. 开发者重新运行后依然报错同样的错误。由于修改已保存，AI 判断是 Python 字节码缓存导致。
3. 执行 `find . -type d -name __pycache__ -exec rm -rf {} +` 清理缓存后，训练成功启动并稳定执行，Loss 稳步下降 (Generator ~0.3, Critic ~0.03)。

---

## 阶段三：配置 5s 推理单 GPU 测试脚本

**开发者**: 请在 `/commondocument/group2/.vscode/launch.json` 为我创建 LongLive 训练和推理单步调试的脚本。
**AI**: 添加了 `LongLive I2V Training`, `LongLive I2V Inference`, 和 `LongLive T2V Inference` 三个 VSCode 调试配置。

**开发者**: 现在我需要使用 `longlive_inference.yaml` 原始的推理脚本、配置和权重，在不使用 LoRA 的情况下，生成时长为 5s 的视频，使用 1 号卡。请给出指令。
**AI**: 
1. 5秒视频（120帧），配置 `num_output_frames: 120` 和 `lora_ckpt: null`。
2. 新建了专门的配置文件 `configs/longlive_inference_5s_no_lora.yaml`。
3. 给出了启动命令：`CUDA_VISIBLE_DEVICES=1 python inference.py --config_path configs/longlive_inference_5s_no_lora.yaml`。

---

## 阶段四：修复 inference.py 命令行参数支持

**开发者**: （运行上述命令时加上了额外的命令行参数）
**报错**: `inference.py: error: unrecognized arguments: --data_path prompts/test_prompts.txt --output_folder outputs/debug_t2v_inference --num_output_frames 21 --num_samples 1`
**AI**: 之前的 `inference.py` 使用的是 `argparse`，并且仅定义了 `--config_path`，不支持从命令行覆盖其他参数。
**解决方案**: 更新代码增加支持：
```python
parser.add_argument("--data_path", type=str, help="Path to prompts file")
parser.add_argument("--output_folder", type=str, help="Directory to save generated videos")
parser.add_argument("--num_output_frames", type=int, help="Number of frames to generate")
parser.add_argument("--num_samples", type=int, help="Number of samples per prompt")
```
并在代码中使用这些参数动态重新赋值覆盖 `OmegaConf.load` 得到的 `config`。

---

## 阶段五：核心代码逻辑 Q&A

**开发者**:
1. `kv_cache_size = local_attn_cfg * self.frame_seq_length` 是什么意思？
2. `all_num_frames = [self.num_frame_per_block] * num_blocks` 
3. `y` 的切片逻辑是对的吗？

**AI**:
1. **`kv_cache_size`**: 计算本地注意力机制（Local Attention Window）所需的 KV Cache Token 总数。`local_attn_cfg` 是帧数（如 12），`frame_seq_length` 是每帧的 Token 数。
2. **`all_num_frames`**: 构建去噪循环的“时间分块列表”。如果 `num_frame_per_block` 为 4，且有 5 个块，列表即为 `[4, 4, 4, 4, 4]`，意味着接下来的 Temporal 循环会每次处理 4 帧。
3. **`y` 的切片逻辑 (`u[:, start:end]`) 验证**: 是正确的。因为 `run_vae_encoder` 处理完成后，去除了 Batch 维度，变成了 `[Channels, Time, Height, Width]`，所以切片操作确实验作在 Time 维度，符合分窗处理对应参考帧的逻辑。

---

## 阶段六：上传 Github 与归档

**开发者**: 当前目录下的文件和 GitHub 上的有什么不同吗？我需要你将代码上传到 GitHub，所有能传递的文件都行。我们历史对话的记录，如果可以，也可以在目录下新建一个文档来存放。
**AI**:
1. 解释了 `inference.py`（命令参数追加）和 `pipeline/causal_inference.py`（增加了大量中文注释）是本地存在但未提交到 GitHub 的差异。
2. 配置了 `.gitignore` 防止大文件及缓存上传。
3. 创建了 `docs/DEVELOPMENT_HISTORY.md` 和本对话记录，一同提交并 Push 到了 GitHub `main` 分支。
