import torch
from stablediffusion.classes.base import BaseModel
from stablediffusion.classes.settings import Txt2ImgArgs
from torch import autocast
from contextlib import nullcontext


class Txt2Img(BaseModel):
    args = Txt2ImgArgs
    current_model = None
    reqtype = "txt2img"

    def current_sample_handler_pass(self, image):
        pass

    def sample(self, options=None, image_handler=None):
        super().sample(options)
        opt = self.opt
        model = self.model
        data = self.data
        sampler = self.plms_sampler
        start_code = self.start_code
        negative_prompt = opt.negative_prompt
        print("NEGATIVE PROMPT: ", negative_prompt)
        self.image_handler = image_handler
        self.current_sampler = sampler
        self.set_seed()
        precision_scope = autocast if opt.precision == "autocast" \
            else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    prompts = data[0]
                    unconditional_conditioning = None
                    if opt.scale != 1.0:
                        unconditional_conditioning = model.get_learned_conditioning(
                            [negative_prompt]
                        )
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    conditioning = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    self.current_model = model
                    samples_ddim, _ = sampler.sample(
                        S=opt.ddim_steps,
                        conditioning=conditioning,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=unconditional_conditioning,
                        eta=opt.ddim_eta,
                        x_T=start_code,
                        image_handler=self.current_sample_handler_pass,
                    )
                    data = self.prepare_image(samples_ddim, True)
                    # self.image_handler(data, options)
                    return data
