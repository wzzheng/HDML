# Hardness-Aware Deep Metric Learning
Implementation of Hardness-Aware Deep Metric Learning (CVPR 2019 Oral) in Tensorflow.

- HDML:  [Hardness-Aware Deep Metric Learning](https://arxiv.org/abs/1903.05503.pdf)

Work in progress.

Please use the citation provided below if it is useful to your research:

Wenzhao Zheng, Zhaodong Chen, Jiwen Lu, and Jie Zhou, Hardness-Aware Deep Metric Learning, arXiv, abs/1903.05503, 2019. 

```bash
@inproceedings{zheng2019hardness,
  title={Hardness-aware deep metric learning},
  author={Zheng, Wenzhao and Chen, Zhaodong and Lu, Jiwen and Zhou, Jie},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={72--81},
  year={2019}
}
```


# Dependencies
```bash
pip install tensorflow==1.10.0
```

# Dataset
Stanford Cars Dataset (Cars196)

-- Download from (https://ai.stanford.edu/~jkrause/cars/car_dataset.html) or use datasets/cars196_downloader.py. 

-- Convert to hdf5 file using cars196_converter.py.

-- Put it in datasets/data/cars196/cars196.hdf5.

# Pretrained model
GoogleNet V1 pretrained model can be downloaded from (https://github.com/Wei2624/Feature_Embed_GoogLeNet)

# Usage
## For Cars196 dataset:

```bash
python main_npair.py --dataSet='cars196' --batch_size=128 --Regular_factor=5e-3 --init_learning_rate=7e-5 --load_formalVal=False --embedding_size=128 --loss_l2_reg=3e-3 --init_batch_per_epoch=500 --batch_per_epoch=64 --max_steps=8000 --beta=1e+4 --lr_gen=1e-2 --num_class=99 --_lambda=0.5 --s_lr=1e-3
```


# Code Reference
deep\_metric\_learning (https://github.com/ronekko/deep_metric_learning) by [ronekko](https://github.com/ronekko) for dataset codes. 
