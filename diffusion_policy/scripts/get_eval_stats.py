import json
import argparse
import os
import numpy as np
from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS



def compute_stats(checkpoint_path):
    stats = dict(
        pretrain=dict(),
        target=dict(),
    )

    assert os.path.exists(checkpoint_path)

    for split in ["pretrain", "target"]:
        split_dir = os.path.join(checkpoint_path, split)
        if not os.path.exists(split_dir):
            continue
        for task_name in os.listdir(split_dir):
            task_dir = os.path.join(split_dir, task_name)
            stats_path = os.path.join(task_dir, "eval_log.json")
            if not os.path.exists(stats_path):
                continue
            with open(stats_path, 'r') as f:
                this_data = json.load(f)
            
            sr_key = f"success_rate/{task_name}"
            if sr_key in this_data:
                stats[split][task_name] = int(this_data[sr_key] * 100)

    all_task_names = set(list(stats["pretrain"].keys()) + list(stats["target"].keys()))
    atomic_task_names = [task for task in all_task_names if task in list(ATOMIC_TASK_DATASETS.keys())]
    composite_task_names = [task for task in all_task_names if task in list(COMPOSITE_TASK_DATASETS.keys())]

    print("ATOMIC TASK EVALS")
    pretrain_vals = []
    target_vals = []
    for task_name in sorted(atomic_task_names):
        pretrain_sr = None
        target_sr = None
        if "pretrain" in stats:
            pretrain_sr = stats["pretrain"].get(task_name)
        if "target" in stats:
            target_sr = stats["target"].get(task_name)
        str_to_print = f"{task_name}: {pretrain_sr} / {target_sr}"
        print(str_to_print)

        if pretrain_sr is not None:
            pretrain_vals.append(pretrain_sr)
        if target_sr is not None:
            target_vals.append(target_sr)
    print("AVG:", np.mean(pretrain_vals), "/", np.mean(target_vals))


    print()
    print()

    print("COMPOSITE TASK EVALS")
    pretrain_vals = []
    target_vals = []
    for task_name in sorted(composite_task_names):
        pretrain_sr = stats["pretrain"].get(task_name)
        target_sr = stats["target"].get(task_name)
        str_to_print = f"{task_name}: {pretrain_sr} / {target_sr}"
        print(str_to_print)

        if pretrain_sr is not None:
            pretrain_vals.append(pretrain_sr)
        if target_sr is not None:
            target_vals.append(target_sr)
    print("AVG:", np.mean(pretrain_vals), "/", np.mean(target_vals))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="relative path to eval dir, eg. 2025.07.23/20.11.51_pretrain_diffusion_transformer_hybrid_human_45atomic_78composite/evals/epoch=0300-target_mean_score=0.400",
    )
    args = parser.parse_args()
    compute_stats(args.dir)