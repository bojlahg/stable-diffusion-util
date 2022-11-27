call conda activate ldm
set prompt=modern disney style
call title %prompt%
call python scripts/img2img_batch.py --prompt "%prompt%" --seed 0 --ddim_eta 0.0 --strength 0.5 --scale 30 --ddim_steps 20 --indir "inputs/restyle/" --outdir "outputs/restyle/" --ckpt "models/ldm/sd/moDi-v1-pruned.ckpt"

pause