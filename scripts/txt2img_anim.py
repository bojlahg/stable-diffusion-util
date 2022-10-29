import argparse, os, sys, glob
from datetime import datetime
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model
    
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res
    
def main():
    opt_ddim_steps = 20
    opt_ddim_eta = 0.0
    opt_plms = False
    opt_fixed_code = True
    opt_width = 512
    opt_height = 512
    opt_c = 4
    opt_f = 8       
    opt_n_samples = 1
    opt_n_iter = 1
    opt_n_rows = 1
    opt_scale_from = -100
    opt_scale_to = 100
    opt_config = "configs/stable-diffusion/v1-inference.yaml"
    opt_precision = "autocast"
    opt_ckpt = "models/ldm/sd/sd-v1-4.ckpt"
    opt_seed_start = 0
    opt_seed_end = opt_seed_start + 1
    opt_from_file = None
    opt_prompt = "Retrofuturism, oil on canvas"
    opt_outdir = f"outputs/anim"
    opt_start_frame = 0
    opt_framecount = 2000
    opt_verbose = False

    config = OmegaConf.load(f"{opt_config}")
    model = load_model_from_config(config, f"{opt_ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt_plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt_outdir, exist_ok=True)
    outpath = opt_outdir

    batch_size = opt_n_samples
    n_rows = opt_n_rows if opt_n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, f"{opt_prompt}_{opt_seed_start}_{opt_scale_from}-{opt_scale_to}")
    os.makedirs(sample_path, exist_ok=True)
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    end_code = None
    if opt_fixed_code:
        seed_everything(opt_seed_start)
        start_code = torch.randn([opt_n_samples, opt_c, opt_height // opt_f, opt_width // opt_f], device=device)

        seed_everything(opt_seed_end)
        end_code = torch.randn([opt_n_samples, opt_c, opt_height // opt_f, opt_width // opt_f], device=device)
    
    precision_scope = autocast if opt_precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope(True):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                prompt = opt_prompt

                for n in trange(opt_n_iter, desc="Sampling"):
                    uc = None
                    if isinstance(prompt, tuple):
                        prompt = list(prompt)
                    c = model.get_learned_conditioning(prompt)
                    shape = [opt_c, opt_height // opt_f, opt_width // opt_f]
                    
                    for frame in range(opt_start_frame, opt_framecount):
                        opt_scale = opt_scale_from + (opt_scale_to - opt_scale_from) * (frame / opt_framecount)
                        print(f"frame = {frame} {opt_scale}\n")
                        #frame_code = slerp(frame / opt_framecount, start_code, end_code)
                        
                        #if opt_scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                        
                        samples_ddim, _ = sampler.sample(S=opt_ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt_n_samples,
                                                         shape=shape,
                                                         verbose=opt_verbose,
                                                         unconditional_guidance_scale=opt_scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt_ddim_eta,
                                                         x_T=start_code)
                        
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, f"{frame:05}.png"))

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
