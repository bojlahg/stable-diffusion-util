call conda activate ldm
set prompt=Emma Watson
call python scripts/txt2img_mod.py --ckpt models/ldm/sd/sd-v1-4.ckpt --seed 542613 --constseed --ddim_eta 0.0 --scale 7.5 --n_samples 1 --n_iter 4 --n_rows 2 --ddim_steps 20 --W 512 --H 512 --f 8 --C 4 --prompt "%prompt%" --style_file="styles/custom.txt"
pause