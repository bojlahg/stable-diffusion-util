call conda activate ldm
set subdir=Robot, by Caravaggio
call python scripts/resample.py --ckpt "models/ldm/bsr_sr2/model.ckpt" --ddim_eta 0.0 --ddim_steps 20 --plms --inpath "outputs\txt2img\%subdir%" --outdir "outputs\supersample\%subdir%"
pause