# LongLive I2V Development & Troubleshooting History

This document archives the key technical discussions, troubleshooting steps, and code changes made during the development of the Image-to-Video (I2V) support for LongLive.

## 1. Key Bug Fixes & Implementations

### 1.1 `WanCLIPEncoder` Dimension Mismatch
**Issue:** `RuntimeError: Tensors must have same number of dimensions: got 2 and 3` during cross-attention context concatenation.
**Fix:** Removed an erroneous `.squeeze(0)` in `WanCLIPEncoder.forward` that was stripping the batch dimension from the CLIP image features. 

### 1.2 `WanI2VCrossAttention` Cache Argument Missing
**Issue:** `TypeError: WanI2VCrossAttention.forward() got an unexpected keyword argument 'crossattn_cache'`
**Fix:** The I2V specific cross-attention module was lacking the `crossattn_cache` argument in its `forward` signature, which is passed automatically by the pipeline. Added the argument and implemented the corresponding K/V caching logic.

### 1.3 `inference.py` CLI Argument Overrides
**Issue:** `unrecognized arguments: --data_path ...` when trying to override config file parameters via command line.
**Fix:** Updated `inference.py`'s `argparse` configuration to explicitly accept `--data_path`, `--output_folder`, `--num_output_frames`, and `--num_samples`, and used them to override the corresponding `OmegaConf` config values at runtime.

### 1.4 Python Cache Issue (`__pycache__`)
**Issue:** After fixing Python code, the trainer still crashed with the same error due to old compiled bytecodes.
**Fix:** Cleared the `__pycache__` and `*.pyc` files using `find` and `rm`, ensuring the latest source code modifications were loaded by the distributed workers.

## 2. Code Logic Q&A

### 2.1 `kv_cache_size = local_attn_cfg * self.frame_seq_length`
**Q: What does this line mean?**
**A:** This calculates the **total number of tokens** needed for the Local Attention Window's Key/Value cache.
- `local_attn_cfg`: The number of frames covered by the local attention mechanism (e.g., 12 frames).
- `self.frame_seq_length`: The number of tokens per frame ($Latent\_Height \times Latent\_Width$).
This size determines the dimension of the K/V buffers initialized in the GPU memory.

### 2.2 `all_num_frames = [self.num_frame_per_block] * num_blocks`
**Q: What is the purpose of this?**
**A:** It constructs a list representing the temporal chunking for the denoising loop. If the model processes blocks of 4 frames, and there are 5 blocks, the list becomes `[4, 4, 4, 4, 4]`. The temporal denoising loop iterates over this list to generate the video block by block.

### 2.3 `y` Slicing Logic (`y=[u[:, start:end] for u in y]`)
**Q: Is the slicing logic for `y` correct?**
**A:** Yes. `run_vae_encoder` returns a list of 4D tensors `[Channels, Time, Height, Width]` (the batch dimension is removed for individual sample processing).
- The first `:` selects all channels (usually 8 channels for I2V, including the mask).
- The `start:end` slices the **Time** dimension.
Thus, this correctly extracts the conditioning latents for the current temporal window being generated.
