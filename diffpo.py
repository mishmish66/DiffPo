# Copyright (c) 2025 Misha Lvovsky
# SPDX-License-Identifier: MIT
#
# Author: Misha Lvovsky https://www.ccs.neu.edu/home/mishmish/
#
# Originally based on denoising_diffusion_pytorch_1d.py from denoising-diffusion-pytorch by lucidrains
# Copyright (c) Phil Wang
# Original license: MIT


# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "beartype",
#     "ema-pytorch",
#     "jaxtyping",
#     "lerobot",
#     "tyro",
# ]
#
# [tool.ruff.lint]
# ignore = ["F722"]
# ///

import math
import random
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from multiprocessing import cpu_count
from pathlib import Path
from random import shuffle
from typing import Literal, cast, final

import numpy as np
import torch as th
import torch.nn.functional as F
import tyro
from beartype import beartype as typechecker
from einops import einsum, pack, rearrange, reduce, unpack
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from jaxtyping import Float, Int, jaxtyped
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from rich import print as rprint
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch import Tensor, nn
from torch.amp.autocast_mode import autocast
from torch.nn import Module, ModuleList
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2

import wandb


@typechecker
@dataclass
class Args:
    # Data config
    repo_id: str
    train_batch_size: int
    test_batch_size = 16
    action_key: str = "actions"
    train_portion: float = 0.9
    seed: int = 42

    max_img_size: int = 224
    images_stack_steps: int = 0
    proprio_hist_steps: int = 3
    action_chunk_steps: int = 20

    data_workers: int = int(cpu_count() * 1.5)

    # Optim params
    gradient_accumulate_every: int = 1
    train_lr: float = 1e-4
    train_num_steps: int = 100000
    ema_decay: float = 0.995

    # Condition Params
    time_dim: int = 128
    img_embed_dim: int = 64

    # UNet Params
    unet_init_dim: int = 64
    u_dims: tuple[int, ...] = (512, 1024, 2048)

    # Conv Params
    conv_groups: int = 8
    dropout: float = 0.0

    # Trainer Params
    save_every: int = 10_000
    wandb_proj: str = "diffpo"


## training loop ##


def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    lrds_meta = LeRobotDatasetMetadata(args.repo_id)

    episode_idxs = list(range(lrds_meta.total_episodes))
    shuffle(episode_idxs)

    split_idx = int(lrds_meta.total_episodes * args.train_portion)
    train_episodes, test_episodes = episode_idxs[:split_idx], episode_idxs[split_idx:]

    period = 1.0 / lrds_meta.fps
    delta_timestamps = {}

    img_keys = []
    prp_keys = []

    for key, info in lrds_meta.features.items():
        if key.startswith("obs"):
            if info["dtype"] == "video":
                img_keys.append(key)
            else:
                prp_keys.append(key)

    def get_prp(batch: dict):
        return th.cat([batch[key] for key in prp_keys], dim=-1)

    def get_img(batch: dict):
        imgs = [batch[img_key] for img_key in img_keys]
        if args.images_stack_steps == 0:
            # Pretend to stack images for consistency
            return rearrange(imgs, "i ... c h w -> ... 1 i c h w")
        return rearrange(imgs, "i ... t c h w -> ... t i c h w")

    delta_timestamps[args.action_key] = np.arange(0, args.action_chunk_steps) * period
    for key in img_keys:
        delta_timestamps[key] = np.arange(-args.images_stack_steps, 1) * period
    for key in prp_keys:
        delta_timestamps[key] = np.arange(-args.proprio_hist_steps, 1) * period

    train_lrds = LeRobotDataset(
        args.repo_id,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
    )

    test_lrds = LeRobotDataset(
        args.repo_id,
        episodes=test_episodes,
        delta_timestamps=delta_timestamps,
    )

    dummy = train_lrds[0]
    dummy_prp = th.cat([dummy[key] for key in prp_keys], dim=-1)
    *_, prp_steps, prp_dim = dummy_prp.shape
    dummy_img = th.stack([dummy[key] for key in img_keys])
    n_cams = len(img_keys)
    act_dim = lrds_meta.features[args.action_key]["shape"][-1]

    vision_backbone = ImgEncoder(args.img_embed_dim)
    # Just send this through once to init lazy modules
    vision_backbone.encode(dummy_img)

    net = Unet(
        channels=act_dim,
        cond_dim=args.time_dim + args.img_embed_dim * n_cams + prp_dim * prp_steps,
        init_dim=args.unet_init_dim,
        u_dims=args.u_dims,
        groups=args.conv_groups,
        dropout=args.dropout,
    )

    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")

    diff_model = DiffPo(
        network=net,
        time_dim=args.time_dim,
        img_encoder=vision_backbone,
        normalizer=Normalizer(act_dim),
        act_dim=act_dim,
        chunk_steps=args.action_chunk_steps,
        prp_dim=prp_dim,
        prp_steps=prp_steps,
        n_cams=n_cams,
        img_steps=args.images_stack_steps + 1,
        sampling_timesteps=32,
    ).to(device)
    ema = EMA(diff_model)

    # Print parameter counts
    total_params = sum(p.numel() for p in diff_model.parameters())
    rprint(f"Total params: {total_params:,}")

    # model
    mp_context = None if args.data_workers < 4 else "forkserver"

    dl = DataLoader(
        repeat_dset(train_lrds, 100),
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.data_workers // 4 * 3,
        multiprocessing_context=mp_context,
    )
    dl = cycle(dl)

    test_dl = DataLoader(
        repeat_dset(test_lrds, 100),
        batch_size=args.test_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.data_workers // 4,
        multiprocessing_context=mp_context,
    )
    test_dl = cycle(test_dl)

    # optimizer
    optim = AdamW(diff_model.parameters(), lr=args.train_lr)

    global_step = 0

    wandb.init(project=args.wandb_proj)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(f"runs/{timestamp}")
    model_dir = save_dir / "models"

    save_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir()

    console = Console()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        prog_train = progress.add_task(
            "[cyan]Training...",
            total=args.train_num_steps,
            completed=global_step,
        )
        while global_step < args.train_num_steps:
            diff_model.train()

            total_loss = 0.0

            for _ in range(args.gradient_accumulate_every):
                batch = next(dl)
                act_y = batch[args.action_key].to(th.float32).to(device)
                prp_x = get_prp(batch).to(th.float32).to(device)
                img_x = get_img(batch).to(device)

                loss = diff_model.p_losses(act_y, img_x, prp_x).mean()
                loss = loss / args.gradient_accumulate_every
                total_loss += loss.item()

                loss.backward()

            optim.step()
            optim.zero_grad()

            with th.no_grad():
                diff_model.eval()
                batch = next(test_dl)

                test_act_y = batch[args.action_key].to(th.float32).to(device)
                test_prp_x = get_prp(batch).to(th.float32).to(device)
                test_img_x = get_img(batch).to(device)

                test_loss = diff_model.p_losses(
                    test_act_y,
                    test_img_x,
                    test_prp_x,
                ).mean()

                if global_step % 100 == 0:
                    cond = diff_model.prepare_cond(test_img_x, test_prp_x)
                    result, noise = diff_model.ddim_sample(cond, test_act_y.shape)
                    ddim_l2 = th.norm(result - test_act_y, dim=-1, p=2).mean()

                    wandb.log({"l2": ddim_l2}, step=global_step)

                diff_model.train()

            progress.update(
                prog_train,
                description=f"[cyan]loss: {total_loss:.4f}, test: {test_loss:.4f}",
                advance=1,
            )
            wandb.log(
                {"loss": total_loss, "test": test_loss},
                step=global_step,
            )

            global_step += 1
            ema.update()

            if global_step % args.save_every == 0:
                th.save(diff_model, model_dir / f"diffpo_{global_step}.pth")
                th.save(ema.ema_model, model_dir / f"ema-diffpo_{global_step}.pth")


## Actual Diffusion Stuff ##


@jaxtyped(typechecker=typechecker)
@dataclass
class ModelPrediction:
    pred_noise: Float[Tensor, "*batch t a"]
    x_start: Float[Tensor, "*batch t a"]


@typechecker
@final
class DiffPo(Module):
    def __init__(
        self,
        network: "Unet",
        time_dim: int,
        img_encoder: "ImgEncoder",
        normalizer: "Normalizer",
        act_dim: int,
        chunk_steps: int,
        prp_dim: int,
        prp_steps: int,
        n_cams: int,
        img_steps: int,
        *,
        denoise_steps=128,
        sampling_timesteps=None,
        beta_schedule: Literal["cosine", "linear"] = "cosine",
        ddim_sampling_eta=0.0,
    ):
        super().__init__()

        self.model = network
        self.time_embedder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.img_encoder = img_encoder

        match beta_schedule:
            case "linear":
                betas = linear_beta_schedule(denoise_steps)
            case "cosine":
                betas = cosine_beta_schedule(denoise_steps)

        alphas = 1.0 - betas
        alphas_cumprod = th.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (denoise_steps,) = betas.shape
        self.num_timesteps = int(denoise_steps)

        # sampling related parameters

        self.sampling_timesteps = (
            sampling_timesteps or denoise_steps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= denoise_steps
        self.is_ddim_sampling = self.sampling_timesteps < denoise_steps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        self.betas = nn.Buffer(betas)
        self.alphas_cumprod = nn.Buffer(alphas_cumprod)
        self.alphas_cumprod_prev = nn.Buffer(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.sqrt_alphas_cumprod = nn.Buffer(th.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = nn.Buffer(th.sqrt(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = nn.Buffer(th.sqrt(1.0 / alphas_cumprod))

        self.sqrt_recipm1_alphas_cumprod = nn.Buffer(th.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.posterior_variance = nn.Buffer(posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.posterior_log_variance_clipped = nn.Buffer(
            th.log(posterior_variance.clamp(min=1e-20)),
        )
        self.posterior_mean_coef1 = nn.Buffer(
            betas * th.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.posterior_mean_coef2 = nn.Buffer(
            (1.0 - alphas_cumprod_prev) * th.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        self.normalizer = normalizer
        self.act_dim = act_dim
        self.chunk_steps = chunk_steps
        self.n_cams = n_cams
        self.img_steps = img_steps
        self.prp_steps = prp_steps
        self.prp_dim = prp_dim

    @jaxtyped(typechecker=typechecker)
    def model_predictions(
        self,
        x: Float[Tensor, "*batch t a"],
        t: Int[Tensor, " d"],
        cond: Float[Tensor, "*batch c"],
    ) -> ModelPrediction:
        t_cond = self.time_embedder(t)
        full_cond = th.cat([t_cond, cond], dim=-1)
        x = rearrange(x, "... t a -> ... a t")
        model_output = self.model(x, full_cond)

        pred_noise = model_output
        x_factor = self.sqrt_recip_alphas_cumprod[t]
        noise_factor = self.sqrt_recipm1_alphas_cumprod[t]
        x_scaled = einsum(x, x_factor, "... a t, ... -> ... a t")
        noise_scaled = einsum(pred_noise, noise_factor, "... a t, ... -> ... a t")

        x_start = (x_scaled - noise_scaled).to(x.dtype)

        return ModelPrediction(
            rearrange(pred_noise, "... a t -> ... t a"),
            rearrange(x_start, "... a t -> ... t a"),
        )

    @jaxtyped(typechecker=typechecker)
    @th.no_grad()
    def ddim_sample(
        self,
        cond: Float[Tensor, "*batch {self.cond_dim}"],
        shape: tuple[int, ...],
    ) -> tuple[
        Float[Tensor, "*batch {self.chunk_steps} {self.act_dim}"],
        Float[Tensor, "*batch {self.chunk_steps} {self.act_dim}"],
    ]:
        device, eta = (self.betas.device, self.ddim_sampling_eta)
        *batch_dims, _ = cond.shape

        times = th.linspace(
            -1,
            self.num_timesteps - 1,
            steps=self.sampling_timesteps + 1,
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))

        noise = th.randn(shape, device=device, dtype=th.float32)
        actions = noise

        for time, time_next in pairwise(times):
            time_cond = th.full(batch_dims, time, device=device, dtype=th.long)
            pred = self.model_predictions(actions, time_cond, cond)

            if time_next < 0:
                actions = pred.x_start
                continue

            alpha = self.alphas_cumprod[time].to(noise.dtype)
            alpha_next = self.alphas_cumprod[time_next].to(noise.dtype)

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = th.randn_like(actions)

            actions = (
                pred.x_start * alpha_next.sqrt() + c * pred.pred_noise + sigma * noise
            )

        actions = self.normalizer.unnormalize(actions)
        return actions, noise

    @autocast("cuda", enabled=False)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)

        x_scale = self.sqrt_alphas_cumprod[t]
        noise_scale = self.sqrt_one_minus_alphas_cumprod[t]

        x_scaled = einsum(x_start, x_scale, "... t a, ... -> ... t a")
        noise_scaled = einsum(noise, noise_scale, "... t a, ... -> ... t a")
        return (x_scaled + noise_scaled).to(x_start.dtype)

    @jaxtyped(typechecker=typechecker)
    def p_losses(
        self,
        x_start: Float[Tensor, "*batch {self.chunk_steps} {self.act_dim}"],
        img: Float[Tensor, "*batch {self.img_steps} {self.n_cams} c h w"],
        prp: Float[Tensor, "*batch {self.prp_steps} {self.prp_dim}"],
        t: Int[Tensor, "*batch"] | None = None,
        noise: Float[Tensor, "*batch {self.chunk_steps} {self.act_dim}"] | None = None,
    ):
        x_start = self.normalizer.normalize(x_start)
        *batch_dims, _, _ = x_start.shape
        device = x_start.device

        if noise is None:
            noise = th.randn_like(x_start)
        if t is None:
            t = th.randint(0, self.num_timesteps, batch_dims, device=device).long()
        # use the noise to do forward diffusion
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # And now learn to reverse the process
        robo_cond = self.prepare_cond(img, prp)
        time_cond = self.time_embedder(t)
        cond = th.cat([time_cond, robo_cond], dim=-1)

        x = rearrange(x, "... t a -> ... a t")
        model_out = self.model(x, cond)
        model_out = rearrange(model_out, "... a t -> ... t a")

        # Compute the loss
        return F.mse_loss(model_out, noise, reduction="none")

    @property
    def cond_dim(self) -> int:
        img_dim = self.img_steps * self.n_cams * self.img_encoder.dout
        prp_dim = self.prp_steps * self.prp_dim
        return img_dim + prp_dim

    @jaxtyped(typechecker=typechecker)
    def prepare_cond(
        self,
        img: Float[Tensor, "*batch {self.img_steps} {self.n_cams} c h w"],
        prp: Float[Tensor, "*batch {self.prp_steps} {self.prp_dim}"],
    ) -> Float[Tensor, "*batch {self.cond_dim}"]:
        img_embeds = self.img_encoder.encode(img)
        img_cond = rearrange(img_embeds, "... s nc f -> ... (s nc f)")
        prp_cond = rearrange(prp, "... t d -> ... (t d)")
        return th.cat([img_cond, prp_cond], dim=-1)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return th.linspace(beta_start, beta_end, timesteps, dtype=th.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = th.linspace(0, timesteps, steps, dtype=th.float64)
    alphas_cumprod = th.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return th.clip(betas, 0, 0.999)


## helper functions ##


def repeat_dset(dset: Dataset, reps: int):
    repeated = dset
    for i in range(reps):
        repeated = dset + repeated

    return repeated


def cycle(dl):
    while True:
        for data in dl:
            yield data


## non-learned Helper Modules ##


class Normalizer(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.max = nn.Buffer(th.ones(dim) * -th.inf)
        self.min = nn.Buffer(th.ones(dim) * th.inf)
        self.dim = dim

    def normalize(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        if self.training:
            x_max = reduce(x, "... t d -> d", "max")
            self.max = self.max.maximum(x_max)
            x_min = reduce(x, "... t d -> d", "min")
            self.min = self.min.minimum(x_min)

        half_range = (self.max - self.min) / 2
        center = (self.max + self.min) / 2
        return (x - center) / (half_range + 1e-3)

    def unnormalize(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        half_range = (self.max - self.min) / 2
        center = (self.max + self.min) / 2
        return x * half_range + center


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(th.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


## UNet ##


@typechecker
class Unet(Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        init_dim: int,
        u_dims: tuple[int, ...],
        groups: int,
        dropout: float,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.init_conv = nn.Conv1d(channels, init_dim, 7, padding=3)
        dims = [init_dim, *u_dims]

        # layers
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(dims) - 1

        for ind, (dim_in, dim_out) in enumerate(pairwise(dims)):
            is_last = ind >= (num_resolutions - 1)
            dblock = DownBlock(
                dim_in,
                dim_out,
                cond_dim,
                downsample=not is_last,
                dropout=dropout,
                groups=groups,
            )
            self.downs.append(dblock)

        mid_dim = dims[-1]
        self.mid_block1 = CondResBlock(
            mid_dim,
            mid_dim,
            mid_dim,
            cond_dim,
            groups=groups,
        )
        self.mid_block2 = CondResBlock(
            mid_dim,
            mid_dim,
            mid_dim,
            cond_dim,
            groups=groups,
        )

        for ind, (dim_in, dim_out) in enumerate(pairwise(reversed(dims))):
            is_last = ind >= (num_resolutions - 1)

            ublock = UpBlock(
                dim_in,
                dim_in,
                dim_out,
                cond_dim,
                upsample=not is_last,
                dropout=dropout,
                groups=groups,
            )
            self.ups.append(ublock)

        self.final_resblock = CondResBlock(
            init_dim * 2,
            init_dim * 2,
            init_dim,
            cond_dim,
            # groups=groups,
            dropout=dropout,
        )
        self.final_conv = nn.Conv1d(init_dim, channels, 1)
        self.dcond = cond_dim

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[Tensor, "*b {self.channels} s"],
        cond: Float[Tensor, "*b {self.dcond}"],
    ) -> Float[Tensor, "*b {self.channels} s"]:
        x = self.init_conv(x)
        r = x.clone()
        h = []

        for downblock in self.downs:
            downblock = cast("DownBlock", downblock)
            x = downblock.resblock1(x, cond)
            h.append(x)

            x = downblock.resblock2(x, cond)
            h.append(x)

            x = downblock.outproj(x)

        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        for upblock in self.ups:
            upblock = cast("UpBlock", upblock)
            x = th.cat([x, h.pop()], dim=1)
            x = upblock.resblock1(x, cond)

            x = th.cat([x, h.pop()], dim=1)
            x = upblock.resblock2(x, cond)

            x = upblock.outproj(x)

        x = th.cat((x, r), dim=1)

        x = self.final_resblock(x, cond)
        return self.final_conv(x)


@typechecker
class Block(Module):
    def __init__(self, dim, dim_out, *, groups: int = 1, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, groups=groups, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.din = dim
        self.dout = dim_out

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[Tensor, "*batch {self.din} l"],
        scale_shift: Float[Tensor, "2 *batch {self.dout}"] | None = None,
    ):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale[..., None] + 1) + shift[..., None]

        x = self.act(x)
        return self.dropout(x)


@typechecker
class CondResBlock(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hid: int,
        dim_out: int,
        cond_dim: int,
        *,
        groups: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block1 = Block(dim_in, dim_hid, groups=groups, dropout=dropout)
        self.block2 = Block(dim_hid, dim_out, groups=groups)
        self.res_conv = (
            nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        )

        self.cond_layer = nn.Sequential(
            nn.Linear(cond_dim, 2 * dim_hid),
            Rearrange("... (f df) -> f ... df", f=2),
        )
        self.dcond = cond_dim
        self.din = dim_in
        self.dout = dim_out

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[Tensor, "*b {self.din} l"],
        cond: Float[Tensor, "*b {self.dcond}"] | None,
    ):
        if cond is not None:
            scale_shift = self.cond_layer(cond)
        else:
            scale_shift = None

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


@typechecker
class DownBlock(Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        cond_dim: int,
        *,
        downsample: bool = True,
        groups: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.resblock1 = CondResBlock(
            dim_in,
            dim_out,
            dim_out,
            cond_dim=cond_dim,
            dropout=dropout,
            groups=groups,
        )
        self.resblock2 = CondResBlock(
            dim_out,
            dim_out,
            dim_out,
            cond_dim=cond_dim,
            dropout=dropout,
            groups=groups,
        )
        if downsample:
            self.outproj = nn.Conv1d(
                dim_out,
                dim_out,
                kernel_size=4,
                stride=2,
                padding=1,
                groups=groups,
            )
        else:
            self.outproj = nn.Conv1d(
                dim_out,
                dim_out,
                kernel_size=3,
                padding=1,
                groups=groups,
            )

        self.din, self.dout = dim_in, dim_out


@typechecker
class UpBlock(Module):
    def __init__(
        self,
        dim_in: int,
        udim: int,
        dim_out: int,
        cond_dim: int,
        upsample: bool,
        groups: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.resblock1 = CondResBlock(
            dim_in + udim,
            dim_in,
            dim_in,
            cond_dim,
            dropout=dropout,
            groups=groups,
        )
        self.resblock2 = CondResBlock(
            dim_in + udim,
            dim_in,
            dim_in,
            cond_dim,
            dropout=dropout,
            groups=groups,
        )
        if upsample:
            self.outproj = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(dim_in, dim_out, 3, padding=1, groups=groups),
            )
        else:
            self.outproj = nn.Conv1d(dim_in, dim_out, 3, padding=1, groups=groups)

        self.din = dim_in
        self.dout = dim_out


### Vision Backbone ###


@typechecker
class ImgEncoder(Module):
    def __init__(self, out_dim: int):
        # Transform
        super().__init__()
        self.img_preprocess = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(th.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.Resize(None, max_size=args.max_img_size),
            ],
        )

        # Initialize Backbone
        backbone_layers = list(resnet18(weights=ResNet18_Weights.DEFAULT).children())
        self.backbone = nn.Sequential(
            *backbone_layers[:-2],
            nn.LazyConv2d(out_channels=out_dim // 2, kernel_size=1),
        )
        self.outproj = nn.Sequential(
            Rearrange("... k p -> ... (k p)"),
            nn.LazyLinear(out_dim),
            nn.ReLU(),
        )
        self.dout = out_dim

    @jaxtyped(typechecker=typechecker)
    def encode(
        self,
        imgs: Float[Tensor, "*batch c h w"],
    ) -> Float[Tensor, "*batch {self.dout}"]:
        x, ps = pack([imgs], "* c h w")
        x = self.img_preprocess(x)

        # Apply the spatial softmax
        x, *_ = unpack(self.backbone(x), ps, "* c h w")
        *_, c, h, w = x.shape
        spatial_weights = F.softmax(rearrange(x, "... c h w -> ... c (h w)"), dim=-1)
        h_space = th.arange(h, device=x.device) / h
        w_space = th.arange(w, device=x.device) / w
        xs, ys = th.meshgrid(h_space, w_space)
        pos_grid = rearrange([xs, ys], "d i j -> d (i j)")

        x = einsum(pos_grid, spatial_weights, "p k, ... w k -> ... k p")
        return self.outproj(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
