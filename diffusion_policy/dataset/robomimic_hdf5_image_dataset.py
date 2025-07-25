from robomimic.utils.dataset import SequenceDataset
from robomimic.macros import LANG_EMB_KEY
from robomimic.utils.dataset import CustomWeightedRandomSampler, MetaDataset
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

from threadpoolctl import threadpool_limits
from diffusion_policy.common.pytorch_util import dict_apply

import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.lang_utils as LangUtils

import torch
import numpy as np
from typing import Dict, List

from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)

from diffusion_policy.dataset.robomimic_replay_image_dataset import normalizer_from_stat


class RobomimicHDF5ImageDataset(SequenceDataset,BaseImageDataset):

    """
    Dataset class for sampling directly from Robomimic HDF5 dataset with image observations.

    Same args as RobomimicReplayImageDataset for now.
    """
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
            val_ratio=0.0, # validation not implemented yet
            filter_key=None,
            action_keys=('actions',),
            lang_encoder=None,
            del_lang_encoder_after_init=False,
        ):

        assert not abs_action, "abs_action not supported"
        obs_dict= shape_meta["obs"].copy()

        # sequence dataset does not consider language embeddings as part of obs
        obs_dict.pop(LANG_EMB_KEY, None)
        obs_keys = obs_dict.keys()

        # convert horizon and n_obs_steps to frame_stack and seq_length
        frame_stack = n_obs_steps
        seq_length = horizon - frame_stack + 1
        dataset_keys = action_keys

        action_config = {action_keys[0]: {'normalization': None}}
        action_config.update({
            "actions":{
                "normalization": None,
            },
            "actions_abs":{
                "normalization": "min_max",
            },
            "action_dict/right_gripper": {
                "normalization": None,
            },
            "action_dict/right_ee_pos_delta": {
                "normalization": None,
            },
            "action_dict/right_ee_ori_delta": {
                "normalization": None,
            },
            "action_dict/right_joint_abs": {
                "normalization": "min_max",
            },
            "action_dict/base": {
                "normalization": None,
            },
            "action_dict/torso": {
                "normalization": None,
            },
            "action_dict/abs_pos": {
                "normalization": "min_max"
            },
            "action_dict/abs_rot_axis_angle": {
                "normalization": "min_max",
                "format": "rot_axis_angle"
            },
            "action_dict/abs_rot_6d": {
                "normalization": None,
                "format": "rot_6d"
            },
            "action_dict/rel_pos": {
                "normalization": None,
            },
            "action_dict/rel_rot_axis_angle": {
                "normalization": None,
                "format": "rot_axis_angle"
            },
            "action_dict/rel_rot_6d": {
                "normalization": None,
                "format": "rot_6d"
            },
            "action_dict/gripper": {
                "normalization": None,
            },
            "action_dict/base_mode": {
                "normalization": None,
            },
        })
        hdf5_normalize_obs = False

        # assert pad_before and pad_after equal to nobs and action length
        # init sequence dataset with kwargs
        SequenceDataset.__init__(self, 
            hdf5_path=dataset_path, 
            obs_keys=obs_keys,
            action_keys=action_keys,
            dataset_keys=dataset_keys, 
            action_config=action_config,
            frame_stack=frame_stack,
            seq_length=seq_length,
            pad_frame_stack=True,
            pad_seq_length=True,
            load_next_obs=False,
            hdf5_normalize_obs=hdf5_normalize_obs,
            filter_by_attribute=filter_key,
            normalize_actions=False, # don't normalize actions in dataset (will be normalized by diffusion policy model later)
            lang_encoder=lang_encoder,
            del_lang_encoder_after_init=del_lang_encoder_after_init,
        )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.action_key = action_keys[0]
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.action_size = self.shape_meta['action']["shape"][0]
        self.filter_key = filter_key


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # threadpool_limits(1)

        # super call to get data
        data = SequenceDataset.__getitem__(self, idx)

        # Rest is same as RobomimicReplayImageDataset
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        # print(data["obs"].keys())
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data["obs"][key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
        for key in self.lowdim_keys:
            obs_dict[key] = data["obs"][key][T_slice].astype(np.float32)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['actions'][:, :self.action_size].astype(np.float32))
        }

        return torch_data
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # Almost same as robomimic_replay_image_dataset.py
        normalizer = LinearNormalizer()

        # action
        # stat = array_to_stats(self._get_all_data(self.action_key).astype(np.float32))
        # if self.abs_action:
        #     if stat['mean'].shape[-1] > 10:
        #         # dual arm
        #         this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
        #     else:
        #         this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
        #     if self.use_legacy_normalizer:
        #         this_normalizer = normalizer_from_stat(stat)
        # else:
        #     # already normalized
        #     this_normalizer = get_identity_normalizer_from_stat(stat)
        # normalizer['action'] = this_normalizer
        action_normalization_stats = self.get_action_normalization_stats()
        scale = np.concatenate([action_normalization_stats[ac_key]["scale"][0] for ac_key in self.action_keys])
        offset = np.concatenate([action_normalization_stats[ac_key]["offset"][0] for ac_key in self.action_keys])
        normalizer['action'] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict={}, #stat
        )

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self._get_all_data("obs/"+key).astype(np.float32))

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == LANG_EMB_KEY:
                # don't normalize language embeddings
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('sin'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('cos'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        self.close_and_delete_hdf5_handle()
        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        return self._get_all_data(self.action_key)[:, :self.action_size]

    def _get_all_data(self, key):
        """
        Get all data for a given key across all episodes.
        """
        if LANG_EMB_KEY in key:
            return np.array(list(self._demo_id_to_demo_lang_emb.values()))

        if self.filter_key is not None:
            demo_keys = [elem.decode("utf-8") for elem in np.array(self.hdf5_file["mask/{}".format(self.filter_key)][:])]
        else:
            demo_keys = self.demos
        
        return np.concatenate([self.hdf5_file["data/{}/{}".format(ep, key)][()] for ep in demo_keys], axis=0)


class RobomimicCotrainingHDF5ImageDataset(MetaDataset, BaseImageDataset):
    """
    Dataset class for sampling for multiple datasets for cotraining. Each dataset is a Robomimic HDF5 dataset.
    Very similar to robomimic MetaDataset.
    TODO: rename
    """
    def __init__(self,
            shape_meta: dict,
            dataset_paths: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0, # validation not implemented yet
            filter_key=None,
            action_keys=('actions',), # assumes that all datasets in the cotraining mixture use the same action keys
            normalize_weights_by_ds_size=False,
        ):

        # get the language encoder once
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        lang_encoder = LangUtils.LangEncoder(
            device=device,
        )

        self.datasets = [
                RobomimicHDF5ImageDataset(
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                horizon=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                n_obs_steps=n_obs_steps,
                abs_action=abs_action,
                rotation_rep=rotation_rep,
                use_legacy_normalizer=use_legacy_normalizer,
                use_cache=use_cache,
                seed=seed,
                # dont validate on sim since we doing real world downstream!
                val_ratio=0,
                action_keys=action_keys,
                filter_key=filter_key,
                lang_encoder=lang_encoder,
                del_lang_encoder_after_init=False,
            ) for dataset_path in dataset_paths
        ]
        
        del lang_encoder # delete the language encoder when done

        self.lowdim_keys = self.datasets[0].lowdim_keys
        self.rgb_keys = self.datasets[0].rgb_keys
        self.abs_action = abs_action
        self.ds_weights = [1/len(self.datasets)]*len(self.datasets)
        self.action_keys = action_keys
        MetaDataset.__init__(self, datasets=self.datasets, ds_weights=self.ds_weights, normalize_weights_by_ds_size=normalize_weights_by_ds_size)

    def get_validation_dataset(self):
        # if val ratio is not 0, then use the real dataset for validation since that is our downstream setting
        return self.datasets[0].get_validation_dataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:

        # Aggregate data from all datasets and normalize together
        normalizer = LinearNormalizer()

        # action
        # stat = array_to_stats(np.concatenate([ds.get_all_actions().astype(np.float32) for ds in self.datasets], axis=0))
        # if self.abs_action:
        #     if stat['mean'].shape[-1] > 10:
        #         # dual arm
        #         this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
        #     else:
        #         this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

        #     if self.use_legacy_normalizer:
        #         this_normalizer = normalizer_from_stat(stat)
        # else:
        #     # already normalized
        #     this_normalizer = get_identity_normalizer_from_stat(stat)
        # normalizer['action'] = this_normalizer
        action_normalization_stats = self.get_action_normalization_stats()
        scale = np.concatenate([action_normalization_stats[ac_key]["scale"][0] for ac_key in self.action_keys])
        offset = np.concatenate([action_normalization_stats[ac_key]["offset"][0] for ac_key in self.action_keys])
        normalizer['action'] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict={}, #stat
        )

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(np.concatenate([ds._get_all_data("obs/" + key).astype(np.float32) for ds in self.datasets], axis=0))

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == LANG_EMB_KEY:
                # don't normalize language embeddings
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('sin'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('cos'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.cat([ds.get_all_actions() for ds in self.datasets], dim=0)


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )