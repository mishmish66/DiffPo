# DiffPo

This is an implementation of the Diffusion Policy from https://arxiv.org/abs/2303.04137 with a focus on simplicity.
It uses lerobot datasets by default, most of them should just work? (if you set the action key)

This implementation is designed to be basic and instructive, as well as easy to copy and paste in and out of projects as needed and modified there in place.
I hope you find this as fun to use as it was for me to make!

## Usage

Run the script with --help
Any PEP 723 compliant tool should work, or a venv with the deps installed of course

```bash
python diffpo.py --repo-id "lerobot/svla_so100_pickplace" --train-batch-size 64 --action-key "action"
```

Feel free to reach out for any questions or help getting this working.

Heavily based on the [lucidrains Diffusion 1d](https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py) implementation inspired by both that repo and the [CleanRL repo](https://github.com/vwxyzjn/cleanrl/tree/master/cleanrl)

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
