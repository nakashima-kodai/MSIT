python -B train.py --model=pix2pixHD --name=pix2pixHD_128p --batch_size=16 --load_size=128 --crop_size=128 --ngf=32 --n_epoch_fix_local=20 --load_pretrain=./ckpt/pix2pix_64p --num_D=3 --load_epoch=200