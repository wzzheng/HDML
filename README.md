# Hardness-Aware Deep Metric Learning
Implementation of Hardness-Aware Deep Metric Learning (CVPR 2019 Oral) in Tensorflow.

Work in progress.

- HDML:  [Hardness-Aware Deep Metric Learning](https://arxiv.org/abs/1903.05503.pdf)

Please use the citation provided below if it is useful to your research:

Wenzhao Zheng, Zhaodong Chen, Jiwen Lu, and Jie Zhou, Hardness-Aware Deep Metric Learning, arXiv, abs/1903.05503, 2019. 

```bash
@article{zheng2019hardness,
  title={Hardness-Aware Deep Metric Learning},
  author={Zheng, Wenzhao and Chen, Zhaodong and Lu, Jiwen and Zhou, Jie},
  journal={arXiv preprint arXiv:1903.05503},
  year={2019}
}
```


# Dependencies
```bash
pip install tensorflow==1.10.0
```

# Dataset
Stanford Cars Dataset (Cars196)

可从官网 (https://ai.stanford.edu/~jkrause/cars/car_dataset.html) 或者使用lib/datasets/cars196_downloader.py文件下载，并使用cars196_converter.py转化为hdf5文件，放在datasets/data/cars196/cars196.hdf5

# Usage
## For Cars196 dataset:

```bash
python CVPR_main_HDML_npair.py --dataSet='cars196' --batch_size=128 --Regular_factor=5e-3 --init_learning_rate=7e-5 --load_formalVal=False --embedding_size=128 --loss_l2_reg=3e-3 --init_batch_per_epoch=500 --batch_per_epoch=64 --max_steps=8000 --beta=1e+4 --lr_gen=1e-2 --num_class=99 --_lambda=0.5 --s_lr=1e-3
```


# Code Reference
deep\_metric\_learning (https://github.com/ronekko/deep_metric_learning) by [ronekko](https://github.com/ronekko) for dataset codes. 
