# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import datasets
import torchvision.transforms.functional as TF



class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TextFolderDataset(Dataset):
    """Dataset for reading text prompts from individual .txt files in a folder."""
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


class ShardingLMDBDataset(Dataset):
    """Dataset for loading sharded LMDB data with images for I2V training.
    
    Expected LMDB format:
    - latents_shape: b'num_items 21 16 60 104' (space-separated string)
    - img_{idx}_data: image bytes
    - prompts_{idx}_data: prompt text bytes
    - latents_{idx}_data: latent bytes (optional)
    """
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []
        self.data_path = data_path

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            if os.path.isdir(path) and fname.startswith('shard'):
                env = lmdb.open(path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
                self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            with env.begin(write=False) as txn:
                latents_shape_bytes = txn.get(b'latents_shape')
                if latents_shape_bytes:
                    # Parse string format: "1250 21 16 60 104"
                    shape_str = latents_shape_bytes.decode('utf-8')
                    self.latents_shape[shard_id] = list(map(int, shape_str.split()))
                else:
                    # Fallback: count img entries
                    cursor = txn.cursor()
                    count = sum(1 for key, _ in cursor if key.startswith(b'img_'))
                    self.latents_shape[shard_id] = [count]
                
            for local_i in range(self.latents_shape[shard_id][0]):
                if len(self.index) < max_pair:
                    self.index.append((shard_id, local_i))

        self.max_pair = max_pair
        print(f"ShardingLMDBDataset: Loaded {len(self.envs)} shards, {len(self.index)} total pairs")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        shard_id, local_idx = self.index[idx]
        env = self.envs[shard_id]
        
        with env.begin(write=False) as txn:
            # Get prompt (key: prompts_{idx}_data)
            prompt_key = f"prompts_{local_idx}_data".encode()
            prompt_bytes = txn.get(prompt_key)
            prompts = prompt_bytes.decode('utf-8') if prompt_bytes else ""

            # Get image (key: img_{idx}_data)
            img_key = f"img_{local_idx}_data".encode()
            img_bytes = txn.get(img_key)
            if img_bytes:
                # Image stored as raw RGB bytes (480, 832, 3)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape((480, 832, 3))
                img = Image.fromarray(img_array)
                img = TF.to_tensor(img).sub_(0.5).div_(0.5)
            else:
                img = torch.zeros(3, 480, 832)

            # Get latents (key: latents_{idx}_data) - optional
            latent_key = f"latents_{local_idx}_data".encode()
            latent_bytes = txn.get(latent_key)
            if latent_bytes:
                # Latents shape: [21, 16, 60, 104] as float16
                latents = np.frombuffer(latent_bytes, dtype=np.float16)
                shape = self.latents_shape[shard_id][1:]
                latents = latents.reshape(shape)
                latents = torch.tensor(latents.copy(), dtype=torch.float32)
                # Assuming T, C, H, W -> C, T, H, W
                if len(shape) == 4:
                    latents = latents.permute(1, 0, 2, 3)
            else:
                latents = None

        result = {
            "prompts": prompts,
            "img": img
        }
        if latents is not None:
            result["ode_latent"] = latents
        
        return result


class TwoTextDataset(Dataset):
    """Dataset that returns two text prompts per sample for prompt-switch training.

    The dataset behaves similarly to :class:`TextDataset` but instead of a single
    prompt, it provides *two* prompts – typically the first prompt is used for the
    first segment of the video, and the second prompt is used after a temporal
    switch during training.

    Args:
        prompt_path (str): Path to a text file containing the *first* prompt for
            each sample. One prompt per line.
        switch_prompt_path (str): Path to a text file containing the *second*
            prompt for each sample. Must have the **same number of lines** as
            ``prompt_path`` so that prompts are paired 1-to-1.
    """
    def __init__(self, prompt_path: str, switch_prompt_path: str):
        # Load the first-segment prompts.
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        # Load the second-segment prompts.
        with open(switch_prompt_path, encoding="utf-8") as f:
            self.switch_prompt_list = [line.rstrip() for line in f]

        assert len(self.switch_prompt_list) == len(self.prompt_list), (
            "The two prompt files must contain the same number of lines so that "
            "each first-segment prompt is paired with exactly one second-segment prompt."
        )

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return {
            "prompts": self.prompt_list[idx],            # first-segment prompt
            "switch_prompts": self.switch_prompt_list[idx],  # second-segment prompt
            "idx": idx,
        }


class MultiTextDataset(Dataset):
    """Dataset for multi-segment prompts stored in a JSONL file.

    Each line is a JSON object, e.g.
        {"prompts": ["a cat", "a dog", "a bird"]}

    Args
    ----
    prompt_path : str
        Path to the JSONL file
    field       : str
        Name of the list-of-strings field, default "prompts"
    cache_dir   : str | None
        ``cache_dir`` passed to HF Datasets (optional)
    """

    def __init__(self, prompt_path: str, field: str = "prompts", cache_dir: str | None = None):
        self.ds = datasets.load_dataset(
            "json",
            data_files=prompt_path,
            split="train",
            cache_dir=cache_dir,
            streaming=False, 
        )

        assert len(self.ds) > 0, "JSONL is empty"
        assert field in self.ds.column_names, f"Missing field '{field}'"

        seg_len = len(self.ds[0][field])
        for i, ex in enumerate(self.ds):
            val = ex[field]
            assert isinstance(val, list), f"Line {i} field '{field}' is not a list"
            assert len(val) == seg_len,  f"Line {i} list length mismatch"

        self.field = field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "idx": idx,
            "prompts_list": self.ds[idx][self.field],  # List[str]
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data
