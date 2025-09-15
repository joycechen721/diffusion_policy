"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from omegaconf import OmegaConf
import copy

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from termcolor import colored
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from robocasa.utils.dataset_registry import get_ds_path
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS, POST_TRAINING_TASKS


ATOMIC_TASKS = list(SINGLE_STAGE_TASK_DATASETS.keys())
DEFAULT_TASKS = POST_TRAINING_TASKS["atomic"] + POST_TRAINING_TASKS["composite"]

def eval_task(checkpoint, base_output_dir, device, task, num_rollouts, num_envs, split, overwrite):
    if base_output_dir is None:
        base_output_dir = os.path.join(os.path.dirname(checkpoint), "../evals", os.path.basename(checkpoint).replace(".ckpt", ""), split)

    output_dir = os.path.join(base_output_dir, task)

    out_path = os.path.join(output_dir, 'eval_log.json')
    if overwrite is False and os.path.exists(out_path):
        # click.confirm(f"Output path {out_path} already exists! Overwrite?", abort=True)
        print(f"Eval stats path {out_path} already exists! Skipping.")
        return

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg = copy.deepcopy(OmegaConf.to_container(cfg))
    cfg["task"]["env_runner"]["env_kwargs"] = {
        "seed": 1111111,
        "use_camera_obs": True,
        "use_object_obs": True,
        "camera_depths": False,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "camera_names": ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'],
        "camera_widths": 256,
        "camera_heights": 256,
        "ignore_done": True,
        "reward_shaping": False,
    }
    if split == "train":
        cfg["task"]["env_runner"]["env_kwargs"].update(
            obj_instance_split="train",
            style_ids=-2,
            layout_ids=-2,
        )
    elif split == "test":
        cfg["task"]["env_runner"]["env_kwargs"].update(
            obj_instance_split="test",
            style_ids=None,
            layout_ids=None,
            layout_and_style_ids=[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]],
            clutter_mode=1,
        )
    elif split == "all":
        cfg["task"]["env_runner"]["env_kwargs"].update(
            obj_instance_split=None,
            style_ids=-3,
            layout_ids=-3,
        )
    else:
        raise ValueError("Invalid split. Choose among train/test/all")
    cfg = OmegaConf.create(cfg)

    ds_path, ds_meta = get_ds_path(task=task, ds_type="human_raw", split=split, return_info=True)
    
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_test = num_rollouts

    # set dataset path and horizon
    cfg.task.dataset_path = ds_path
    cfg.task.env_runner.dataset_path = ds_path
    cfg.task.dataset.dataset_path = ds_path
    cfg.task.env_runner.max_steps = int(ds_meta["horizon"] * 1.5)
    cfg.task.env_runner.n_envs = num_envs

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    try_num = 1
    MAX_TRIES = 5
    while try_num <= MAX_TRIES:
        env_runner = None
        runner_log = None
        try:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=output_dir)
            runner_log = env_runner.run(policy)
        except Exception as e:
            print(f"Excpetion in env_runner (try {try_num})")
            print(e)
            print()
            try_num += 1
            continue
        
        break
    
    # dump log to json
    if runner_log is not None:
        json_log = dict()
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                json_log[key] = value._path
            else:
                json_log[key] = value
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    # close and delete everything
    if env_runner is not None:
        env_runner.close()
    del policy
    del workspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', default=None)
@click.option('-d', '--device', default='cuda:0')
@click.option('-t', '--tasks', multiple=True, default=DEFAULT_TASKS)
@click.option('-n', '--num_rollouts', default=50)
@click.option('-e', '--num_envs', default=10)
@click.option('-s', '--split', required=True)
# @click.option('--overwrite', is_flag=True, help='Overwrite existing evals.')
def main(checkpoint, output_dir, device, tasks, num_rollouts, num_envs, split): #, overwrite):
    if len(tasks) == 1 and tasks[0] == "atomic_seen":
        tasks = POST_TRAINING_TASKS["atomic_seen"]
    elif len(tasks) == 1 and tasks[0] == "composite_seen":
        tasks = POST_TRAINING_TASKS["composite_seen"]
    elif len(tasks) == 1 and tasks[0] == "composite_unseen":
        tasks = POST_TRAINING_TASKS["composite_unseen"]
    
    for task_i, task in enumerate(tasks):
        print(colored(f"[{task_i+1}/{len(tasks)}] running evals for {task}", "yellow"))
        eval_task(checkpoint, output_dir, device, task, num_rollouts, num_envs, split, overwrite=False)

if __name__ == '__main__':
    main()