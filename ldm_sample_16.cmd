call conda activate ldm
call python scripts/sample_diffusion.py -r models/ldm/pokemon/diffusion_pytorch_model.bin -n 4 --batch_size 4 -c 50 --eta 1
pause