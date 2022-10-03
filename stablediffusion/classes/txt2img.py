import time
import torch
from stablediffusion.classes.base import BaseModel
from torch import autocast
from tqdm import tqdm, trange
from contextlib import nullcontext
from einops import rearrange
import json
from PIL import Image
import numpy as np
import io

class Txt2Img(BaseModel):
    args = [
        {
            "arg": "prompt",
            "type": str,
            "nargs": "?",
            "default": "a painting of a virus monster playing guitar",
            "help": "the prompt to render"
        },
        {
            "arg": "outdir",
            "type": str,
            "nargs": "?",
            "help": "dir to write results to",
            "default": "outputs/txt2img-samples"
        },
        {
            "arg": "skip_grid",
            "action": "store_true",
            "help": "do not save a grid, only individual samples. Helpful when evaluating lots of samples",
        },
        {
            "arg": "skip_save",
            "action": "store_true",
            "help": "do not save individual samples. For speed measurements.",
        },
        {
            "arg": "ddim_steps",
            "type": int,
            "default": 50,
            "help": "number of ddim sampling steps",
        },
        {
            "arg": "plms",
            "action": "store_true",
            "help": "use plms sampling",
        },
        {
            "arg": "laion400m",
            "action": "store_true",
            "help": "uses the LAION400M model",
        },
        {
            "arg": "fixed_code",
            "action": "store_true",
            "help": "if enabled, uses the same starting code across samples ",
        },
        {
            "arg": "ddim_eta",
            "type": float,
            "default": 0.0,
            "help": "ddim eta (eta=0.0 corresponds to deterministic sampling",
        },
        {
            "arg": "n_iter",
            "type": int,
            "default": 2,
            "help": "sample this often",
        },
        {
            "arg": "H",
            "type": int,
            "default": 512,
            "help": "image height, in pixel space",
        },
        {
            "arg": "W",
            "type": int,
            "default": 512,
            "help": "image width, in pixel space",
        },
        {
            "arg": "C",
            "type": int,
            "default": 4,
            "help": "latent channels",
        },
        {
            "arg": "f",
            "type": int,
            "default": 8,
            "help": "downsampling factor",
        },
        {
            "arg": "n_samples",
            "type": int,
            "default": 3,
            "help": "how many samples to produce for each given prompt. A.k.a. batch size",
        },
        {
            "arg": "n_rows",
            "type": int,
            "default": 0,
            "help": "rows in the grid (default: n_samples)",
        },
        {
            "arg": "scale",
            "type": float,
            "default": 7.5,
            "help": "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        },
        {
            "arg": "from-file",
            "type": str,
            "help": "if specified, load prompts from this file",
        },
        {
            "arg": "config",
            "type": str,
            "default": "configs/stable-diffusion/v1-inference.yaml",
            "help": "path to config which constructs model",
        },
        {
            "arg": "ckpt",
            "type": str,
            "default": "models/ldm/stable-diffusion-v1/model.ckpt",
            "help": "path to checkpoint of model",
        },
        {
            "arg": "seed",
            "type": int,
            "default": 42,
            "help": "the seed (for reproducible sampling)",
        },
        {
            "arg": "precision",
            "type": str,
            "help": "evaluate at this precision",
            "choices": ["full", "autocast"],
            "default": "autocast"
        },
    ]

    def sample(self, options=None):
        super().sample(options)
        torch.cuda.empty_cache()
        opt = self.opt
        model = self.model
        sample_path = self.sample_path
        data = self.data
        batch_size = self.batch_size
        sampler = self.plms_sampler
        start_code = self.start_code
        base_count = self.base_count
        self.set_seed()
        precision_scope = nullcontext
        if opt.precision == "autocast":
            precision_scope = autocast
        saved_files = []
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    prompts = data[0]
                    unconditional_conditioning = None
                    if opt.scale != 1.0:
                        unconditional_conditioning = model.get_learned_conditioning(
                            batch_size * [""]
                        )
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                    samples_ddim, _ = sampler.sample(
                        S=opt.ddim_steps,
                        conditioning=c,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=unconditional_conditioning,
                        eta=opt.ddim_eta,
                        x_T=start_code
                    )
                    imgdata = self.prepare_image(model, samples_ddim)
                    print("IMAGE DATA PREPARED, RETURNING")
                    return imgdata

    def prepare_image(self, model, samples_ddim):
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        for x_sample in x_samples_ddim:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            return x_sample
