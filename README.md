# DiffPo

This is an implementation of the Diffusion Policy from https://arxiv.org/abs/2303.04137 with a focus on simplicity.
It uses lerobot datasets by default, most of them should just work? (if you set the action key)

This implementation is designed to be basic and instructive, as well as easy to copy and paste in and out of projects as needed and modified there in place.
I hope you find this as fun to use as it was for me to make!

## Installation

No installation needed! Uses inline script metadata (PEP 723).
Just run with `uv` or your favorite PEP 723-compatible runner.

## Usage

```bash
uv run diffpo.py --repo-id "lerobot/svla_so100_pickplace" --train-batch-size 64 --action-key "action"
```

Feel free to reach out for any questions or help getting this set up.

:)

```bibtex
@article{Chi2023DiffusionPV,
    title   = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
    author  = {Cheng Chi and Siyuan Feng and Yilun Du and Zhenjia Xu and Eric A. Cousineau and Benjamin Burchfiel and Shuran Song},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2303.04137},
    url     = {https://api.semanticscholar.org/CorpusID:257378658}
}
```
