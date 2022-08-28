call conda activate ldm
set prompt=Emma Watson
set style=, by Vincent van Gogh
call python scripts/txt2img_mod.py --ckpt models/ldm/sd/sd-v1-4.ckpt --ddim_eta 0.0 --scale 7.5 --n_samples 1 --n_iter 16 --n_rows 4 --ddim_steps 20 --W 512 --H 768 --f 8 --C 4 --prompt "%prompt%%style%"
pause