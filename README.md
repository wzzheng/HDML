# CVPR2019
The codes for CVPR 2019
## For Cars196 dataset:

python NIPS_main_HNG_npair.py --dataSet='cars196' --batch_size=128 --Regular_factor=5e-3 --init_learning_rate=7e-5 --load_formalVal=False --embedding_size=128 --loss_l2_reg=3e-3 --init_batch_per_epoch=500 --batch_per_epoch=64 --max_steps=12000 --beta=1e+4 --lr_gen=1e-2 --num_class=99 --_lambda=0.5 --s_lr=1e-3

lr decay at 5.5k 6.5k stop at 8k, take the average of last 4 epochs
