"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import os
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import torch

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Top-level toggle for device override.
# Set to "mps" to force Apple Metal, "cpu" to force CPU, or None to use config value.
DEVICE_OVERRIDE = "mps"

def _resolve_device_override(cfg: OmegaConf) -> None:
    # Env override wins over constant.
    requested = os.environ.get("DP_DEVICE", DEVICE_OVERRIDE)
    if requested in (None, "", "none", "null"):
        return
    if not hasattr(cfg, "training") or "device" not in cfg.training:
        return
    normalized = str(requested).lower()
    if normalized.startswith("mps"):
        if torch.backends.mps.is_available():
            cfg.training.device = "mps"
        else:
            print("[train.py] MPS requested but not available; falling back to CPU.")
            cfg.training.device = "cpu"
        return
    cfg.training.device = requested

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    _resolve_device_override(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
