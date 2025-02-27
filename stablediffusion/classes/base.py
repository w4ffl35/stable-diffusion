import io
import argparse
import base64
import os
import torch
import numpy as np
import cv2
import logging
from PIL import Image
from einops import rearrange
from imwatermark import WatermarkEncoder
from torch import autocast
from pytorch_lightning import seed_everything  # FAILS
from omegaconf import OmegaConf
from contextlib import  nullcontext
from stablediffusion.ldm.models.diffusion.ddim import DDIMSampler
from stablediffusion.ldm.models.diffusion.plms import PLMSSampler
from stablediffusion.ldm.util import instantiate_from_config
from itertools import islice

# load safety model
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


class BaseModel:
    """
    Base model class which is inherited by model classes (see classes.txt2img.py and classes.img2img.py)
    :attribute args: list of arguments to be parsed by argparse, set this in each child class
    :attribute opt: parsed arguments, set by parse_arguments()
    :attribute parser: argparse parser, set by parse_arguments()
    :attribute config: config file, set by load_config()
    :attribute data: data to be used for sampling, set by prepare_data()
    :attribute batch_size: batch size, set by prepare_data()
    :attribute n_rows: number of rows in the output image, set by prepare_data()

2    :attribute sampler: sampler object, set by initialize_sampler()
    :attribute device: device to use, set by load_model()
    :attribute model: model object, set by load_model()
    :attribute base_count: base count, set by initialize_base_count()
    :attribute grid_count: grid count, set by initialize_grid_count()
    :attribute start_code: start code, set by initialize_start_code()
    :attribute precision_scope: precision scope, set by set_precision_scope()
    :attribute initialized: whether the model has been initialized, set by initialize()
    """
    args = []
    current_sampler = None
    safety_feature_extractor = None
    safety_checker = None

    def __init__(self, *args, **kwargs):
        self.args = kwargs.get("args", self.args)
        self.opt = {}
        self.initialize_logging()
        self.parser = None
        self.config = None
        self.data = None
        self.batch_size = None
        self.n_rows = None
        self.outpath = None
        self.device = kwargs.get("device", None)
        self.model = kwargs.get("model", None)
        self.wm_encoder = None
        self.base_count = None
        self.grid_count = None
        self.sample_path = None
        self.start_code = None
        self.precision_scope = None
        self.initialized = False
        self.model_base_path = kwargs.get("model_base_path", "")
        self.ckpt = kwargs.get("ckpt", None)
        self.load_safety_model()
        self.init_model(kwargs.get("options", {}))

    def load_safety_model(self):
        safety_model_id = os.path.join(self.model_base_path, "CompVis/stable-diffusion-safety-checker")
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    def load_model_from_config(self, config, ckpt, verbose=False):
        logging.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            logging.info(f"Global Step: {pl_sd['global_step']}")
        model = instantiate_from_config(config.model, self.model_base_path)
        sd = pl_sd["state_dict"]
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            logging.error("missing keys:")
            logging.error(m)
        if len(u) > 0 and verbose:
            logging.error("unexpected keys:")
            logging.error(u)
        model.cuda().half()
        model.eval()
        return model


    def initialize_logging(self):
        # create path and file if not exist
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    @property
    def plms_sampler(self):
        return PLMSSampler(self.model)

    @property
    def ddim_sampler(self):
        return DDIMSampler(self.model)

    def initialize(self):
        """
        Initialize the model
        :return:
        """
        if self.initialized:
            return
        self.initialized = True
        self.load_config()
        if not self.model or not self.device:
            self.load_model()
        self.initialize_start_code()

    def parse_arguments(self):
        """
        Parse arguments, the arguments are defined by each child class
        :return:
        """
        parser = argparse.ArgumentParser()
        for arg in self.args:
            parser.add_argument(
                f'--{arg["arg"]}',
                **{k: v for k, v in arg.items() if k != "arg"}
            )
        self.parser = parser
        self.opt = self.parser.parse_args()

    def parse_options(self, options):
        """
        Parse options
        :param options: options to parse
        :return:
        """
        self.options = options
        for key,val in options.items():
            self.opt.__setattr__(key, val)

    def initialize_options(self):
        """
        Initialize options, by default check for laion400m and set the corresponding options
        :return:
        """
        pass

    def set_seed(self):
        """
        Seed everything using the current seed.
        This allows us to re-seed the model with a new seed that can remain static or be modified, e.g. when sampling.
        :return:
        """
        seed_everything(self.opt.seed)

    def load_config(self):
        """
        Load config file
        :return:
        """
        self.config = OmegaConf.load(f"{self.opt.config}")

    def load_model(self):
        """
        Load the stable diffusion model
        :return:
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = self.load_model_from_config(self.config, f"{self.ckpt}")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def check_safety(self, x_image):
        x_image = x_image.cpu().permute(0, 2, 3, 1).numpy()
        safety_checker_input = self.safety_feature_extractor(self.numpy_to_pil(x_image), return_tensors="pt")
        x_checked_image, has_nsfw_concept = self.safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        for i in range(len(has_nsfw_concept)):
            if has_nsfw_concept[i]:
                x_checked_image[i] = load_replacement(x_checked_image[i])
        x_checked_image = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
        return x_checked_image, has_nsfw_concept

    def filter_nsfw_content(self, x_samples_ddim):
        """
        Check if the samples are safe for work, replace them with a placeholder if not
        :param x_samples_ddim:
        :return:
        """
        if self.opt.do_nsfw_filter:
            x_samples_ddim, has_nsfw = self.check_safety(x_samples_ddim)
        return x_samples_ddim

    def prepare_data(self):
        """
        Prepare data for sampling
        :return:
        """
        batch_size = self.opt.n_samples if "n_samples" in self.opt else 1
        n_rows = self.opt.n_rows if (
                "n_rows" in self.opt and self.opt.n_rows > 0) else 0
        from_file = self.opt.from_file if "from_file" in self.opt else False
        if not from_file:
            prompt = self.opt.prompt if "prompt" in self.opt else None
            data = [batch_size * [prompt]]

        else:
            self.log.info(f"reading prompts from {from_file}")
            with open(from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))
        self.n_rows = n_rows
        self.batch_size = batch_size
        self.data = data

    def initialize_start_code(self):
        """
        Initialize the start based on fixed_code settings
        :return:
        """
        if self.opt.fixed_code:
            self.start_code = torch.randn([
                self.opt.n_samples,
                self.opt.C,
                self.opt.H // self.opt.f,
                self.opt.W // self.opt.f
            ], device=self.device)

    def set_precision_scope(self):
        """
        Define the precision scope
        :return:
        """
        self.precision_scope = autocast if self.opt.precision=="autocast" else nullcontext

    def get_first_stage_sample(self, model, samples):
        samples_ddim = model.decode_first_stage(samples)
        return torch.clamp((samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

    def init_model(self, options):
        self.parse_arguments()
        self.set_seed()
        self.initialize_options()
        if options:
            self.parse_options(options)
        self.prepare_data()
        self.initialize()

    def sample(self, options=None):
        """
        Sample from the model
        :param options:
        :return:
        """
        self.init_model(options)
        self.set_precision_scope()
        self.base_count = len(os.listdir(self.sample_path))
        self.grid_count = len(os.listdir(self.outpath)) - 1

    reqtype = None
    def current_sample_handler(self, img):
        if self.opt.fast_sample:
            data = self.prepare_image_fast(img)
        else:
            data = self.prepare_image(img)
        options = self.options
        options["reqtype"] = self.reqtype
        self.image_handler(data, options)

    def prepare_image(self, samples_ddim, finalize=False):
        x_samples_ddim = self.current_model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        # Do safety check here
        if finalize:
            x_samples_ddim = self.filter_nsfw_content(x_samples_ddim)

        return self.prepare_image_fast(x_samples_ddim)

    # create the same function as prepare_image, but make it faster
    # by not using the model to decode the first stage
    def prepare_image_fast(self, samples_ddim):
        x_sample = samples_ddim[0]
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        return x_sample
