import os
import h5py
import numpy as np
import random
import json
import math
from copy import deepcopy
from contextlib import contextmanager
from collections import OrderedDict
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.pytorch_util import dict_apply
import robomimic.utils.torch_utils as TorchUtils
from tqdm import tqdm
import robomimic.utils.tensor_utils as TensorUtils
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
import copy

import robomimic.utils.lang_utils as LangUtils
from robomimic.macros import LANG_EMB_KEY

import torch.utils.data
import torch
from typing import Dict, List


from gr00t.data.dataset import LeRobotSingleDataset, LE_ROBOT_MODALITY_FILENAME, ModalityConfig, LE_ROBOT_EPISODE_FILENAME
import pathlib

def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values,
    maintaining the order: video, state, action, annotation
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    # Initialize dictionary with ordered keys
    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict

class LerobotDataset(LeRobotSingleDataset, BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0
        ):

        assert n_obs_steps and n_obs_steps > 0
        self.abs_action = abs_action
        assert not self.abs_action, "abs_action is not supported in LerobotDataset"
        dataset_path = pathlib.Path(dataset_path)
        delta_indices = list(range(-n_obs_steps+1, horizon - n_obs_steps + 1))
        delta_indices_obs = list(range(-n_obs_steps+1, 1))
        assert len(delta_indices_obs) == n_obs_steps, \
            f"delta_indices_obs length {len(delta_indices_obs)} != n_obs_steps {n_obs_steps}"   
        modality_keys_dict = get_modality_keys(dataset_path)
        video_modality_keys = modality_keys_dict["video"]
        language_modality_keys = modality_keys_dict["annotation"]
        state_modality_keys = modality_keys_dict["state"]
        action_modality_keys = modality_keys_dict["action"]
        state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]
        modality_configs = {
            "video": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=video_modality_keys,  # we will include all video modalities
            ),
            "state": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=state_modality_keys,
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=action_modality_keys,
            ),
        }

        
        LeRobotSingleDataset.__init__(
            self,
            dataset_path=dataset_path,
            embodiment_tag="oxe_droid",
            modality_configs=modality_configs,
        )
        rgb_keys = dict()
        lowdim_keys = dict()
        obs_shape_meta = copy.deepcopy(shape_meta['obs'])
        self.lang_emb = obs_shape_meta.pop('lang_emb', None)
        if self.lang_emb is not None:
            assert language_modality_keys, "Language modality keys should not be empty if lang_emb is defined"
            self._get_lang_embeddings()
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
        self.action_info = self.shape_meta['action']
        self.lerobot_action_keys = self.action_info['lerobot_keys']
        self.action_size = self.action_info['shape'][0]
    
    def _get_lang_embeddings(self):
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        self._lang_encoder = LangUtils.LangEncoder(
                device=device,
        )
        self._demo_id_to_demo_lang_emb = {}
    
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        id2remark = {e["episode_index"]: e["remarks"] for e in episode_metadata}
        
        for ep_batch in tqdm(np.array_split(self.trajectory_ids, int(math.ceil(len(self.trajectory_ids) / 64)))):
            # get language embedding
            lang_batch = [id2remark[ep] for ep in ep_batch]
            emb_batch = self._lang_encoder.get_lang_emb(lang_batch)
            emb_batch = TensorUtils.to_numpy(emb_batch)
            for batch_idx, ep in enumerate(ep_batch):
                self._demo_id_to_demo_lang_emb[ep] = emb_batch[batch_idx]
        del self._lang_encoder
        self._lang_encoder = None
            
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # threadpool_limits(1)

        # super call to get data
        data = LeRobotSingleDataset.__getitem__(self, idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key, lerobot_keys in self.rgb_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            print(data[lerobot_key].shape)
            obs_dict[key] = np.moveaxis(data[lerobot_key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
        for key, lerobot_keys in self.lowdim_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            obs_dict[key] = data[lerobot_key][T_slice].astype(np.float32)

        if self.lang_emb is not None:
            trajectory_id, _ = self.all_steps[idx]
            lang_emb = self._demo_id_to_demo_lang_emb[trajectory_id]
            obs_dict[LANG_EMB_KEY] = np.tile(
                lang_emb,
                (self.n_obs_steps, 1)
            ).astype(np.float32)
        
        action_concat = []

        for lr_key in self.lerobot_action_keys:
            if lr_key in data:
                action_concat.append(data[lr_key])
            else:
                raise ValueError(f"Key {lr_key} not found in data")
        

        action_concat = np.concatenate(action_concat, axis=-1)
        assert action_concat.shape[-1] == self.action_size, \
            f"action_concat shape mismatch: {action_concat.shape[-1]} != {self.action_size}"
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action_concat.astype(np.float32))
        }

        return torch_data
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # Almost same as robomimic_replay_image_dataset.py
        normalizer = LinearNormalizer()
        assert not self.abs_action, "normalizer for abs_action is not supported in LerobotDataset"


        scale = np.ones((self.action_size), dtype=np.float32)
        offset = np.zeros((self.action_size), dtype=np.float32)
        normalizer['action'] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict={}, #stat
        )

        # obs
        for key, lerobot_keys in self.lowdim_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            # strip "state." prefix
            lerobot_key = lerobot_key.replace("state.", "")
            stat = self._metadata.statistics.state[lerobot_key].model_dump()
            for k, v in stat.items():
                if type(v) is np.ndarray:
                    stat[k] = v.astype(np.float32)

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('sin'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('cos'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer
        
        # lang_emb
        if self.lang_emb is not None:
            dim = int(np.prod(self.lang_emb["shape"]))  
            scale  = np.ones((dim,), dtype=np.float32)  
            offset = np.zeros((dim,), dtype=np.float32) 
            normalizer[LANG_EMB_KEY] = SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict={}, #stat
            )

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer
