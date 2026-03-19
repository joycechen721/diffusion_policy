import os
import h5py
import numpy as np
import random
import json
import math
from copy import deepcopy
from contextlib import contextmanager
from collections import OrderedDict
from diffusion_policy.diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseLowdimDataset
from diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
import robomimic.utils.torch_utils as TorchUtils
from tqdm import tqdm
import robomimic.utils.tensor_utils as TensorUtils
from diffusion_policy.diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
import copy
import torch.utils.data
import torch
from typing import Dict, List
import pathlib

# Handle potential missing robomimic macros
try:
    from robomimic.macros import LANG_EMB_KEY
except ImportError:
    LANG_EMB_KEY = "lang_emb"

from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
from robocasa.utils.groot_utils.groot_dataset import (
    LeRobotSingleDataset, 
    LE_ROBOT_MODALITY_FILENAME, 
    ModalityConfig, 
    LE_ROBOT_EPISODE_FILENAME, 
    LeRobotMixtureDataset
)

def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict

# --- IMAGE/VISUAL DATASET CLASS ---
class LerobotDataset(LeRobotSingleDataset, BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            filter_key=None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            lang_encoder=None,
            del_lang_encoder_after_init=True,
        ):

        assert n_obs_steps and n_obs_steps > 0
        self.abs_action = abs_action
        dataset_path = pathlib.Path(dataset_path)
        delta_indices = list(range(-n_obs_steps+1, horizon - n_obs_steps + 1))
        delta_indices_obs = list(range(-n_obs_steps+1, 1))
        
        modality_keys_dict = get_modality_keys(dataset_path)
        video_modality_keys = modality_keys_dict["video"]
        state_modality_keys = modality_keys_dict["state"]
        action_modality_keys = modality_keys_dict["action"]
        state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]
        
        modality_configs = {
            "video": ModalityConfig(delta_indices=delta_indices_obs, modality_keys=video_modality_keys),
            "state": ModalityConfig(delta_indices=delta_indices_obs, modality_keys=state_modality_keys),
            "action": ModalityConfig(delta_indices=delta_indices, modality_keys=action_modality_keys),
        }

        LeRobotSingleDataset.__init__(
            self, dataset_path=dataset_path, filter_key=filter_key,
            embodiment_tag="oxe_droid", modality_configs=modality_configs,
        )
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths
        
        rgb_keys = dict()
        lowdim_keys = dict()
        obs_shape_meta = copy.deepcopy(shape_meta['obs'])
        self.lang_emb = obs_shape_meta.pop('lang_emb', None)
        
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys[key] = attr["lerobot_keys"]
            elif type == 'low_dim':
                lowdim_keys[key] = attr["lerobot_keys"]
        
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.lerobot_action_keys = self.shape_meta['action']['lerobot_keys']
        self.action_size = self.shape_meta['action']['shape'][0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = LeRobotSingleDataset.__getitem__(self, idx)
        T_slice = slice(self.n_obs_steps)
        obs_dict = dict()
        
        for key, lerobot_keys in self.rgb_keys.items():
            obs_dict[key] = np.moveaxis(data[lerobot_keys[0]][T_slice], -1, 1).astype(np.float32) / 255.
        
        for key, lerobot_keys in self.lowdim_keys.items():
            obs_dict[key] = data[lerobot_keys[0]][T_slice].astype(np.float32)

        action_concat = np.concatenate([data[lr_key] for lr_key in self.lerobot_action_keys], axis=-1)
        
        return {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action_concat.astype(np.float32))
        }

# --- OPTIMIZED LOW-DIM DATASET CLASS WITH RAM CACHING ---
class LerobotLowdimDataset(LeRobotSingleDataset, BaseLowdimDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            filter_key=None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=False,
            use_cache=True, # Default to True for Colab A100/RAM speedup
            seed=42,
            val_ratio=0.0,
            split="train",
        ):

        assert n_obs_steps and n_obs_steps > 0
        self.abs_action = abs_action
        dataset_path = pathlib.Path(dataset_path)
        delta_indices = list(range(-n_obs_steps+1, horizon - n_obs_steps + 1))
        delta_indices_obs = list(range(-n_obs_steps+1, 1))

        modality_keys_dict = get_modality_keys(dataset_path)
        state_modality_keys = [k for k in modality_keys_dict["state"] if k != "state.dummy_tensor"]
        
        modality_configs = {
            "state": ModalityConfig(delta_indices=delta_indices_obs, modality_keys=state_modality_keys),
            "action": ModalityConfig(delta_indices=delta_indices, modality_keys=modality_keys_dict["action"]),
        }

        LeRobotSingleDataset.__init__(
            self, dataset_path=dataset_path, filter_key=filter_key,
            embodiment_tag="oxe_droid", modality_configs=modality_configs,
        )

        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.lerobot_action_keys = self.shape_meta['action']['lerobot_keys']
        self.action_size = self.shape_meta['action']['shape'][0]

        # Handle Train/Val Split
        if val_ratio > 0:
            rng = np.random.default_rng(seed)
            traj_ids = np.array(self.trajectory_ids)
            perm = rng.permutation(len(traj_ids))
            num_val = max(1, int(round(len(traj_ids) * val_ratio)))
            keep_idx = perm[:num_val] if split == "val" else perm[num_val:]
            
            self._trajectory_ids = traj_ids[keep_idx]
            self._trajectory_lengths = np.array(self.trajectory_lengths)[keep_idx]
            self._all_steps = []
            for tid, tlen in zip(self._trajectory_ids, self._trajectory_lengths):
                for t in range(int(tlen)):
                    self._all_steps.append((int(tid), t))
        
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

        lowdim_keys = dict()
        obs_shape_meta = copy.deepcopy(shape_meta['obs'])
        obs_shape_meta.pop('lang_emb', None)
        for key, attr in obs_shape_meta.items():
            if attr.get('type', 'low_dim') == 'low_dim':
                lowdim_keys[key] = attr["lerobot_keys"]
        self.lowdim_keys = lowdim_keys
        self.lowdim_key_order = list(lowdim_keys.keys())

        # --- RAM CACHING ---
        self.use_cache = use_cache
        if self.use_cache:
            print(f"🚀 [Speedup] Pre-loading dataset into RAM ({split})...")
            self.obs_cache = []
            self.action_cache = []
            for i in tqdm(range(len(self._all_steps)), desc="Caching Parquet data"):
                item = self.get_uncached_item(i)
                self.obs_cache.append(item['obs'])
                self.action_cache.append(item['action'])
            
            self.obs_cache = torch.stack(self.obs_cache)
            self.action_cache = torch.stack(self.action_cache)
            print(f"✅ Cache complete. Ready for high-speed GPU training.")

    def get_uncached_item(self, idx: int):
        data = LeRobotSingleDataset.__getitem__(self, idx)
        T_slice = slice(self.n_obs_steps)
        obs_parts = [data[self.lowdim_keys[key][0]][T_slice].astype(np.float32) for key in self.lowdim_key_order]
        obs = np.concatenate(obs_parts, axis=-1)
        action_concat = np.concatenate([data[lr_key] for lr_key in self.lerobot_action_keys], axis=-1)
        return {'obs': torch.from_numpy(obs), 'action': torch.from_numpy(action_concat.astype(np.float32))}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_cache:
            return {'obs': self.obs_cache[idx], 'action': self.action_cache[idx]}
        return self.get_uncached_item(idx)

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        
        # Action Normalizer (Manual Scale/Offset)
        normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
            scale=np.ones((self.action_size), dtype=np.float32),
            offset=np.zeros((self.action_size), dtype=np.float32),
            input_stats_dict={},
        )

        # Obs Normalizer
        obs_scales, obs_offsets = [], []
        stats_out = {k: [] for k in ["min", "max", "mean", "std", "q01", "q99"]}
        
        for key in self.lowdim_key_order:
            lerobot_key = self.lowdim_keys[key][0].split(".")[-1]
            try:
                stat = self._metadata.statistics.state[lerobot_key].model_dump()
            except KeyError:
                # Fallback for keys missing in stats (e.g. augmented features)
                dim = self.shape_meta['obs'][key]['shape'][0]
                obs_scales.append(np.ones(dim, dtype=np.float32))
                obs_offsets.append(np.zeros(dim, dtype=np.float32))
                continue

            for k, v in stat.items():
                stat[k] = np.array(v, dtype=np.float32)

            this_norm = get_range_normalizer_from_stat(stat) if not key.endswith('quat') else get_identity_normalizer_from_stat(stat)
            obs_scales.append(this_norm.params_dict['scale'].detach().cpu().numpy())
            obs_offsets.append(this_norm.params_dict['offset'].detach().cpu().numpy())
            for s_name in stats_out: stats_out[s_name].append(stat[s_name])

        stats_out = {k: np.concatenate(v, axis=0) for k, v in stats_out.items() if v}
        normalizer['obs'] = SingleFieldLinearNormalizer.create_manual(
            scale=np.concatenate(obs_scales, axis=0),
            offset=np.concatenate(obs_offsets, axis=0),
            input_stats_dict=stats_out,
        )
        return normalizer