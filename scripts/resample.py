import argparse, os, sys, glob
import PIL
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange, repeat
import torch, torchvision
import time

from ldm.util import ismap
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


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
    
def get_cond(selected_path):
    example = dict()
    up_f = 4
    c = Image.open(selected_path)
    c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
    c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True)
    c_up = rearrange(c_up, '1 c h w -> 1 h w c')
    c = rearrange(c, '1 c h w -> 1 h w c')
    c = 2. * c - 1.

    c = c.to(torch.device("cuda"))
    example["LR_image"] = c
    example["image"] = c_up

    return example 
    
def run(model, selected_path, ddim_steps, ddim_eta, plms, resize_enabled=False, classifier_ckpt=None, global_step=None):

    example = get_cond(selected_path)

    save_intermediate_vid = False
    n_runs = 1
    masked = False
    guider = None
    ckwargs = None
    mode = 'ddim'
    ddim_use_x0_pred = False
    temperature = 1.
    make_progrow = True
    custom_shape = None

    height, width = example["image"].shape[1:3]
    split_input = height >= 128 and width >= 128

    if split_input:
        ks = 128
        stride = 64
        vqf = 4  #
        model.split_input_params = {"ks": (ks, ks), "stride": (stride, stride),
                                    "vqf": vqf,
                                    "patch_distributed_vq": True,
                                    "tie_braker": False,
                                    "clip_max_weight": 0.5,
                                    "clip_min_weight": 0.01,
                                    "clip_max_tie_weight": 0.5,
                                    "clip_min_tie_weight": 0.01}
    else:
        if hasattr(model, "split_input_params"):
            delattr(model, "split_input_params")

    invert_mask = False

    x_T = None
    for n in range(n_runs):
        if custom_shape is not None:
            x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
            x_T = repeat(x_T, '1 c h w -> b c h w', b=custom_shape[0])

        logs = make_convolutional_sample(example, model, plms=plms,
                                         mode=mode, custom_steps=ddim_steps,
                                         eta=ddim_eta, swap_mode=False, masked=masked,
                                         invert_mask=invert_mask, quantize_x0=False,
                                         custom_schedule=None, decode_interval=10,
                                         resize_enabled=resize_enabled, custom_shape=custom_shape,
                                         temperature=temperature, noise_dropout=0.,
                                         corrector=guider, corrector_kwargs=ckwargs, x_T=x_T, save_intermediate_vid=save_intermediate_vid,
                                         make_progrow=make_progrow,ddim_use_x0_pred=ddim_use_x0_pred
                                         )
    return logs


@torch.no_grad()
def convsample_ddim(model, cond, steps, shape, eta=1.0, plms = False, callback=None, normals_sequence=None,
                    mask=None, x0=None, quantize_x0=False, img_callback=None,
                    temperature=1., noise_dropout=0., score_corrector=None,
                    corrector_kwargs=None, x_T=None, log_every_t=None
                    ):

    if plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    bs = shape[0]  # dont know where this comes from but wayne
    shape = shape[1:]  # cut batch dim
    print(f"Sampling with eta = {eta}; steps: {steps}")
    samples, intermediates = sampler.sample(steps, batch_size=bs, shape=shape, conditioning=cond, callback=callback,
                                         normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta,
                                         mask=mask, x0=x0, temperature=temperature, verbose=False,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs, x_T=x_T)

    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(batch, model, mode="vanilla", plms = False, custom_steps=None, eta=1.0, swap_mode=False, masked=False,
                              invert_mask=True, quantize_x0=False, custom_schedule=None, decode_interval=1000,
                              resize_enabled=False, custom_shape=None, temperature=1., noise_dropout=0., corrector=None,
                              corrector_kwargs=None, x_T=None, save_intermediate_vid=False, make_progrow=True,ddim_use_x0_pred=False):
    log = dict()

    z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                        return_first_stage_outputs=True,
                                        force_c_encode=not (hasattr(model, 'split_input_params')
                                                            and model.cond_stage_key == 'coordinates_bbox'),
                                        return_original_cond=True)

    log_every_t = 1 if save_intermediate_vid else None

    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    z0 = None

    log["input"] = x
    log["reconstruction"] = xrec

    if ismap(xc):
        log["original_conditioning"] = model.to_rgb(xc)
        if hasattr(model, 'cond_stage_key'):
            log[model.cond_stage_key] = model.to_rgb(xc)

    else:
        log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
        if model.cond_stage_model:
            log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
            if model.cond_stage_key =='class_label':
                log[model.cond_stage_key] = xc[model.cond_stage_key]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        img_cb = None

        sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape, plms=plms,
                                                eta=eta,
                                                quantize_x0=quantize_x0, img_callback=img_cb, mask=None, x0=z0,
                                                temperature=temperature, noise_dropout=noise_dropout,
                                                score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                                x_T=x_T, log_every_t=log_every_t)
        t1 = time.time()

        if ddim_use_x0_pred:
            sample = intermediates['pred_x0'][-1]

    x_sample = model.decode_first_stage(sample)

    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        log["sample_noquant"] = x_sample_noquant
        log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
    except:
        pass

    log["sample"] = x_sample
    log["time"] = t1 - t0

    return log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default="outputs/supersample/",
        help="dir to write results to"
    )
    parser.add_argument(
        "--inpath",
        type=str,
        nargs="?",
        default="outputs/txt2img-samples/samples/",
        help="dir to read images from"
    )
    """
    parser.add_argument(
        "--upscale",
        type=int,
        default=4,
        help="upscaling factor",
    )
    """
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    ) 
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/ldm/bsr_sr/config.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/bsr_sr/model.ckpt",
        help="path to checkpoint of model",
    ) 

    opt = parser.parse_args() 
    
    os.makedirs(opt.outdir, exist_ok=True) 

    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt)
    
    if os.path.isfile(opt.inpath):
        filelist = [opt.inpath]
    else:
        filelist = glob.glob(os.path.join(opt.inpath, "*.png"))
    

    for inputfile in filelist:
        print(f"Input file: {inputfile}\n")
        logs = run(model, inputfile, opt.ddim_steps, opt.ddim_eta, opt.plms)
        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2. * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        Image.fromarray(sample[0]).save(os.path.join(opt.outdir, os.path.basename(inputfile)))

if __name__ == "__main__":
    main()
