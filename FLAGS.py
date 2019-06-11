import tensorflow as tf
import numpy as np
import os
import time

# Flags
FLAGS = tf.flags.FLAGS

# Flags for the Basic Model
tf.flags.DEFINE_string('dataSet', 'cars196', 'Training on which dataset, cars196, cub200, products')
# We use NCA Loss in this program
tf.flags.DEFINE_string('LossType', 'NpairLoss', 'The type of Loss to be used in training')
tf.flags.DEFINE_string('log_save_path', './tensorboard_log/', 'Directory to save tenorboard log files')
tf.flags.DEFINE_string('formerTimer', '02-07-14-27/model.ckpt-27900',
                       'The time that the former checkpoint is created')
tf.flags.DEFINE_string('checkpoint_path', './formerTrain/', 'Directory to restore and save checkpoints')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size, 128 is recommended for cars196')
tf.flags.DEFINE_float('Regular_factor', 5e-3,
                      'weight decay factor, we recommend 5e-3 for cars196 and 1e-3 for cub200')
tf.flags.DEFINE_float('init_learning_rate', 7e-5,
                      'initial learning rate, we recommend 7e-5 for cars196 and 6e-5 for cub200')
tf.flags.DEFINE_integer('default_image_size', 227, 'The size of input images')
tf.flags.DEFINE_bool('SaveVal', True, 'Whether save checkpoint')
tf.flags.DEFINE_bool('normalize', True, 'Whether use batch normalization')
tf.flags.DEFINE_bool('load_formalVal', False, 'Whether load former value before training')
tf.flags.DEFINE_float('embedding_size', 128,
                      'The size of embedding, we recommend 128 for cars196 and 256 for cub200')
tf.flags.DEFINE_float('loss_l2_reg', 3e-3,
                      'The factor of embedding l2_loss, we recommend 3e-3 for cars196 and 1.5e-2 for cub200')
tf.flags.DEFINE_integer('init_batch_per_epoch', 500, 'init_batch_per_epoch, 500 for cars and cub)
tf.flags.DEFINE_integer('batch_per_epoch', 64,
                        'The number of batches per epoch, in most situation, '
                        'we recommend 64 for cars196 and 46 for cub200 while 500 for test')
tf.flags.DEFINE_integer('max_steps', 8000, 'The maximum step number')

# Flags for the HDML method
tf.flags.DEFINE_bool('Apply_HDML', True, 'Whether to apply hard-aware Negative Generation')
tf.flags.DEFINE_float('Softmax_factor', 1e+4, 'The weight factor of softmax')
tf.flags.DEFINE_float('beta', 1e+4, 'The factor of negneg, 1e+4 for cars196, 5e+3 for other')
tf.flags.DEFINE_float('lr_gen', 1e-2, '1e-2 for others')
tf.flags.DEFINE_float('alpha', 90, 'The factor in the pulling function')
tf.flags.DEFINE_integer('num_class', 99, 'Number of classes in dataset, 99 for cars, 101 for cub,'
                        '11319 for products')
tf.flags.DEFINE_float('_lambda', 0.5, 'The trade_off between the two part of gen_loss, 0.5 for cars196')
tf.flags.DEFINE_float('s_lr', 1e-3, 'The learning rate of softmax trainer, 1e-3 for cars196')


# To approximately reduce the mean of input images
image_mean = np.array([123, 117, 104], dtype=np.float32)  # RGB
# To shape the array image_mean to (1, 1, 1, 3) => three channels
image_mean = image_mean[None, None, None, [2, 1, 0]]

neighbours = [1, 2, 4, 8, 16, 32]
products_neighbours = [1, 10, 1000]

# Different losses need different method to create batches
if FLAGS.LossType == "Contrastive_Loss":
    method = "pair"
elif FLAGS.LossType == "NpairLoss" or FLAGS.LossType == "AngularLoss" or FLAGS.LossType == "NCA_loss":
    method = "n_pairs_mc"
elif FLAGS.LossType == "Triplet":
    method = 'triplet'
else:
    method = "clustering"
print("method: "+method)

# Using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
