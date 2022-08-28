call conda activate ldm
REM call python scripts/resample.py --ddim_eta 0.0 --ddim_steps 20 --plms --inpath "outputs\txt2img-samples\samples\02507.png" 
call python scripts/resample.py --ddim_eta 0.0 --ddim_steps 20 --plms --inpath "outputs\supersample\01987.png" --outdir "outputs\supersample2"
pause