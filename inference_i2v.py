import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    target_size = (480, 832)  # Hardcoded for LongLive
    img = img.resize((target_size[1], target_size[0]), Image.BICUBIC)
    img = TF.to_tensor(img).sub_(0.5).div_(0.5)
    return img

def load_text(text_path):
    with open(text_path, 'r') as f:
        return f.read().strip()

def run_inference(pipeline, image_path, prompt, args, config, device, clip_fea=None, y=None):
    # 1. Image
    raw_img = load_image(image_path).unsqueeze(0).to(device, dtype=torch.bfloat16)

    # 2. Encode Image Conditions
    with torch.no_grad():
        # clip_fea will be computed inside the loop (or passed in fixed)
        pass
        
        # local context: y [B, 4, 16, 60, 104] (with mask channel)
        y_list = pipeline.vae.run_vae_encoder(raw_img)
        y = y_list[0].to(dtype=torch.bfloat16)
        
        if hasattr(pipeline.generator.model, 'img_emb'):
             # Explicitly cast to avoid mismatch
             pipeline.generator.model.img_emb.to(device=device, dtype=torch.bfloat16)
             
             # Debug Check
             try:
                 ln = pipeline.generator.model.img_emb.proj[0]
                 print(f"Model img_emb LayerNorm weight dtype: {ln.weight.dtype}")
             except Exception as e:
                 print(f"Could not inspect img_emb: {e}")

    # 3. Text Prompt
    prompts = [prompt] * args.num_samples
    
    # 4. Noise
    # config.num_training_frames is typically 21. We generate 20.
    noise_shape = [args.num_samples, 20, 16, 60, 104]
    sampled_noise = torch.randn(noise_shape, device=device, dtype=torch.bfloat16)

    print(f"Generating {args.num_samples} videos for input: {os.path.basename(image_path)}")
    
    # Streaming / Autoregressive Loop
    # Align with LongLive config: Use num_output_frames to determine windows
    # Training window is 20 generated frames + 1 ref
    
    if args.num_windows > 1:
        # Manual override
        num_windows = args.num_windows
        print(f"Manual override: Generating {num_windows} windows")
    else:
        # Config driven
        num_output_frames = getattr(config, "num_output_frames", 21)
        # (Total - 1) / 20. E.g. (81-1)/20 = 4. (21-1)/20 = 1.
        num_windows = max(1, (num_output_frames - 1) // 20)
        print(f"Config driven: num_output_frames={num_output_frames} -> {num_windows} windows")
    
    # Pre-compute reference latent
    # encode_to_latent returns [B, T, C, H, W]
    ref_latent_pure = pipeline.vae.encode_to_latent(raw_img.unsqueeze(2).to(device))
    
    # Store full video sequence
    full_video_latents_list = [ref_latent_pure.to(torch.bfloat16)]
    
    current_ref_img = raw_img
    
    for w_idx in range(num_windows):
        print(f"Generating Window {w_idx+1}/{num_windows}...")
        
    # Compute initial CLIP feature (Global Semantic / ID)
    # CRITICAL Optimization: Keep clip_fea fixed to the input image (Ground Truth).
    # This prevents semantic drift / ID loss that happens if we re-encode generated frames.
    with torch.no_grad():
        clip_fea_fixed = pipeline.image_encoder(raw_img).to(dtype=torch.bfloat16)

    # Initialize Cache State
    # Only need to initialize if num_windows > 1, but safe to always do it via pipeline logic
    current_kv_cache = None
    current_crossattn_cache = None
    global_frame_offset = 0

    for w_idx in range(num_windows):
        print(f"Generating Window {w_idx+1}/{num_windows}...")
        
        # Update Spatial Condition (y) based on previous frame
        # In first window, we compute y from raw_img
        # In subsequent windows, we update y based on `current_ref_img` (last generated frame)
        # to ensure spatial/motion continuity.
        if w_idx == 0:
            current_ref_img_for_y = raw_img
        else:
            current_ref_img_for_y = current_ref_img

        with torch.no_grad():
            y_list = pipeline.vae.run_vae_encoder(current_ref_img_for_y)
            y = y_list[0].to(dtype=torch.bfloat16)

        with torch.no_grad():
            gen_latents = pipeline.inference(
                noise=sampled_noise, 
                text_prompts=prompts,
                return_latents=True,
                clip_fea=clip_fea_fixed.repeat(args.num_samples, 1, 1), # Always use fixed GT CLIP feature
                y=y.repeat(args.num_samples, 1, 1, 1, 1),
                kv_cache=current_kv_cache,
                crossattn_cache=current_crossattn_cache,
                start_frame_idx=global_frame_offset
            )
            _, gen_latents = gen_latents
            
            # Update cache state (pipeline updates self.kv_cache1 in-place, but better to fetch it)
            current_kv_cache = pipeline.kv_cache1
            current_crossattn_cache = pipeline.crossattn_cache
            
            # Update global offset (each window generates 20 frames)
            global_frame_offset += 20
            
            # Resample noise for next iteration (if needed)
            sampled_noise = torch.randn(noise_shape, device=device, dtype=torch.bfloat16)

        # gen_latents: [B, T, C, H, W]
        # Append to full list
        full_video_latents_list.append(gen_latents.to(torch.bfloat16))
        
        # Update Reference: Last frame of this generation
        # gen_latents is [B, 20, 16, 60, 104]
        last_frame_latent = gen_latents[:, -1:, :, :, :] # [B, 1, C, H, W]
        
        # We need pixel space for CLIP/VAE re-encoding?
        # VAE encode is conditioned on pixel input.
        # So we must decode the last frame to use it as ref for next window.
        last_frame_pixel = pipeline.vae.decode_to_pixel(last_frame_latent) # [B, 1, 3, H, W]
        current_ref_img = last_frame_pixel.squeeze(1) # [B, 3, H, W] (Remove Time dim)
        
        # Ensure correct range/dtype for encoder
        # decode_to_pixel returns [-1, 1] usually? 
        # utils/wan_wrapper.py: clamp(-1, 1) is done in decode_to_pixel?
        # Let's check: decode_to_pixel logic does .float().clamp_(-1, 1). 
        # But wait, in inference_i2v.py previously: video = (video * 0.5 + 0.5).clamp(0, 1)
        # So pipeline output is [-1, 1].
        # WanVAEWrapper.run_vae_encoder expects input?
        # load_image does: .sub_(0.5).div_(0.5) -> [-1, 1] range.
        # So `current_ref_img` (from decode) is already mostly in [-1, 1].
        # We can pass it directly.
        current_ref_img = current_ref_img.to(device, dtype=torch.bfloat16)

    # Concatenate all latents along time
    # [Ref, Win1, Win2, ...]
    full_latents = torch.cat(full_video_latents_list, dim=1)
    
    # Ensure latents are bfloat16 for correct VAE decoding
    full_latents = full_latents.to(dtype=torch.bfloat16)
    
    print(f"Decoding full video of length {full_latents.shape[1]} frames...")
    # Decode in chunks to save VRAM if needed, but for now full decode
    # VAE might OOM on long video. Use chunk decoding if available.
    if hasattr(pipeline.vae, 'decode_to_pixel_chunk'):
        video = pipeline.vae.decode_to_pixel_chunk(full_latents, chunk_size=40)
    else:
        video = pipeline.vae.decode_to_pixel(full_latents)
        
    video = (video * 0.5 + 0.5).clamp(0, 1)

    # Save
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for i in range(args.num_samples):
        # video[i] is [T, C, H, W] -> need [T, H, W, C]
        v = video[i].permute(0, 2, 3, 1).cpu()
        if v.dtype != torch.uint8:
            v = (v * 255).to(torch.uint8)
            
        save_path = os.path.join(config.output_folder, f"{base_name}_len{video.shape[1]}.mp4")
        write_video(save_path, v, fps=16)
        print(f"Saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    # Make image_path optional if test_root is provided
    parser.add_argument("--image_path", type=str, default=None, help="Path to input image (single mode)")
    parser.add_argument("--test_root", type=str, default=None, help="Path to test directory containing 'image' and 'prompt' subdirs")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt (overrides config/file in single mode)")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_windows", type=int, default=1, help="Number of streaming windows (iterations)")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    
    # ------------------ Distributed Setup ------------------
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", str(local_rank)))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group("nccl")
        set_seed(config.seed + local_rank)
    else:
        local_rank = 0
        device = torch.device("cuda")
        set_seed(config.seed)
    
    print(f'[Rank {local_rank}] Free VRAM {get_cuda_free_memory_gb(device)} GB')

    # ------------------ Pipeline Init ------------------
    if not hasattr(config, "i2v"):
        config.i2v = True
        
    pipeline = CausalInferencePipeline(config, device=device)
    
    # Load Generator Checkpoint
    # Load Generator Checkpoint
    if config.resume_ckpt:
        ckpt_path = os.path.join(config.resume_ckpt, "model.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(config.resume_ckpt, "generator_ema.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(config.resume_ckpt, "generator.pt")
            
        print(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        
        # Extract generator state
        if "generator_ema" in state_dict:
            print("Found 'generator_ema' key in checkpoint")
            raw_gen_state_dict = state_dict["generator_ema"]
        elif "generator" in state_dict:
            print("Found 'generator' key in checkpoint")
            raw_gen_state_dict = state_dict["generator"]
        elif "model" in state_dict:
            raw_gen_state_dict = state_dict["model"]
        else:
            raw_gen_state_dict = state_dict

        # Clean keys (FSDP/DDP wrappers)
        def _clean_key(name: str) -> str:
            name = name.replace("_fsdp_wrapped_module.", "")
            if name.startswith("module."):
                name = name[7:]
            return name

        cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
        m, u = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        print(f"Checkpoint loaded. Missing: {len(m)}, Unexpected: {len(u)}")
        if len(m) > 0: print(f"Top missing keys: {m[:5]}")
    
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.text_encoder.to(device=device)
    if pipeline.image_encoder:
        pipeline.image_encoder.to(device=device)

    os.makedirs(config.output_folder, exist_ok=True)

    # ------------------ Input Processing Loop ------------------
    
    if args.test_root:
        # Batch Mode
        print(f"Batch mode: Scanning {args.test_root}")
        image_dir = os.path.join(args.test_root, "image")
        prompt_dir = os.path.join(args.test_root, "prompt")
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)])
        
        for img_file in tqdm(image_files, desc="Inferencing"):
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(image_dir, img_file)
            
            # Find corresponding prompt
            prompt_path = os.path.join(prompt_dir, f"{base_name}.txt")
            if os.path.exists(prompt_path):
                prompt = load_text(prompt_path)
            else:
                print(f"Warning: No prompt file found for {img_file}, using default config prompt")
                prompt = config.prompt if config.prompt else "high quality video"
                
            run_inference(pipeline, img_path, prompt, args, config, device)
            
    elif args.image_path:
        # Single Mode
        prompt = args.prompt if args.prompt else config.prompt
        if not prompt: prompt = "high quality video"
        run_inference(pipeline, args.image_path, prompt, args, config, device)
    else:
        print("Error: Must provide either --test_root or --image_path")
        return

if __name__ == "__main__":
    main()
