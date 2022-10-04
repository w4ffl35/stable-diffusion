import numpy as np
import torch
from torch import autocast
from contextlib import nullcontext
from stablediffusion.classes.base import BaseModel
from einops import repeat
from stablediffusion.classes.settings import Img2ImgArgs


class Img2Img(BaseModel):
    args = Img2ImgArgs

    def sample(self, options=None, image_handler=None):
        print("INSIDE IMG2IMG SAMPLE")
        super().sample(options)
        torch.cuda.empty_cache()
        opt = self.opt
        batch_size = 1
        model = self.model
        sampler = self.ddim_sampler
        data = self.data
        device = self.device
        self.image_handler = image_handler

        # convert opt.init_img to tensor from base64
        init_image = self.load_image(opt.init_img)
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

        assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(opt.strength * opt.ddim_steps)
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast if opt.precision == "autocast" else nullcontext

        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    prompts = data[0]
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning([""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    z_enc = sampler.stochastic_encode(
                        init_latent,
                        torch.tensor([t_enc]).to(device))
                    self.current_model = model
                    samples = sampler.decode(
                        z_enc, c, t_enc,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        image_handler=self.current_sample_handler,
                    )
                    if self.opt.fast_sample:
                        data = self.prepare_image(samples, True)
                        self.image_handler(data)
                    return True

    def load_image(self, image):
        """
        :param image: a one dimensional array of a 512x512x3 rgb image (0-255)
        :return:
        """
        # convert image [255, 255, 255, ...] to base64
        image = np.array(image).reshape(512, 512, 3)

        # convert the image to torch tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).cuda().half()

        # convert the image to base64
        image = 2.*image - 1.
        return image
