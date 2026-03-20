# Diffusion Policy

This is the Diffusion Policy fork repo for running RoboCasa benchmark experiments.
This fork is based on the original Diffusion Policy code, hosted at [https://github.com/real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy).

## Modifications for CS188
We modify files in LeRobot dataset configurations in `diffusion_policy/config` to process the training data into a lower state dimension. We specifically use the `diffusion_policy/config/train_diffusion_unet_lowdim.yaml` configuration to train our policy.

## Recommended system specs
For training we recommend a GPU with at least 24 Gb of memory, but 48 Gb+ is prefered.
For inference we recommend a GPU with at least 8 Gb of memory.

## Installation
```
git clone https://github.com/robocasa-benchmark/diffusion_policy
cd diffusion_policy
pip install -e .
```

## Key files
- Training: [train.py](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/train.py)
- Evaluation: [eval_robocasa.py](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/eval_robocasa.py)

## Experiment workflow
```
# train model
python train.py \
--config-name=train_diffusion_transformer_bs192 \
task=robocasa/<dataset-soup>

# Evaluate model
python eval_robocasa.py \
--checkpoint <checkpoint-path> \
--task_set <task-set> \
--split <split>

# Report evaluation results
python diffusion_policy/scripts/get_eval_stats.py \
--dir <outputs-dir>
```
