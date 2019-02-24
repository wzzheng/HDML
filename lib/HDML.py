from FLAGS import *
from lib.nn_Ops import distance, weight_variable, bias_variable
import os


def Pulling(Loss_type, embedding, Jm):
    if Loss_type == 'NpairLoss':
        embedding_split = tf.split(embedding, 2, axis=0)
        anc = embedding_split[0]
        pos = embedding_split[1]
        neg = pos

        anc_tile = tf.reshape(tf.tile(anc, [1, int(FLAGS.batch_size / 2)]), [-1, int(FLAGS.embedding_size)])
        pos_tile = tf.reshape(tf.tile(pos, [1, int(FLAGS.batch_size / 2)]), [-1, int(FLAGS.embedding_size)])
        neg_tile = tf.tile(neg, [FLAGS.batch_size / 2, 1])

        neg2_tile = anc_tile + tf.multiply(
            (neg_tile - anc_tile),
            tf.tile(
                ((distance(anc_tile, pos_tile) + (distance(anc_tile, neg_tile) - distance(anc_tile, pos_tile)) * tf.exp(
                    -FLAGS.alpha / Jm)) / distance(anc_tile, neg_tile)), [1, int(FLAGS.embedding_size)]
            )
        )
        # if D_ap is larger than D_an, the element will be True(1)
        neg_mask = tf.greater_equal(distance(anc_tile, pos_tile), distance(anc_tile, neg_tile))
        op_neg_mask = tf.logical_not(neg_mask)
        neg_mask = tf.cast(neg_mask, tf.float32)
        op_neg_mask = tf.cast(op_neg_mask, tf.float32)
        neg_tile = tf.multiply(neg_tile, neg_mask) + tf.multiply(neg2_tile, op_neg_mask)
        embedding_z_quta = tf.concat([anc, neg_tile], axis=0)
        return embedding_z_quta

    elif Loss_type == 'triple-loss':
        embedding_split = tf.split(embedding, 3, axis=0)
        anc = embedding_split[0]
        pos = embedding_split[1]
        neg = embedding_split[2]
        neg2 = anc + tf.multiply(
            (neg - anc),
            tf.tile(
                ((distance(anc, pos) + (distance(anc, neg) - distance(anc, pos)) * tf.exp(
                    -FLAGS.alpha / Jm)) / distance(anc, neg)), [1, FLAGS.embedding_size]
            )
        )
        # if D_ap is larger than D_an, the element will be True(1)
        neg_mask = tf.greater_equal(distance(anc, pos), distance(anc, neg))
        op_neg_mask = tf.logical_not(neg_mask)
        neg_mask = tf.cast(neg_mask, tf.float32)
        op_neg_mask = tf.cast(op_neg_mask, tf.float32)
        neg = tf.multiply(neg, neg_mask) + tf.multiply(neg2, op_neg_mask)

        # embedding2 is the pulled embedding
        embedding_z_quta = tf.concat([anc, pos, neg], axis=0)
        return embedding_z_quta

    else:
        print("Your loss type is not suit for HDML")
        os._exit()


def npairSplit(embedding, Loss_type=FLAGS.LossType, size=1024):
    if Loss_type == 'NpairLoss':
        embedding_yp = tf.slice(input_=embedding, begin=[0, 0], size=[FLAGS.batch_size, size])
        embedding_yq = tf.slice(
            input_=embedding, begin=[FLAGS.batch_size, 0],
            size=[int(FLAGS.batch_size/2+np.square(FLAGS.batch_size/2)), size])
        return embedding_yp, embedding_yq
    else:
        print("Not n-pair-loss")


def cross_entropy(embedding, label, size=1024):
    with tf.variable_scope("Softmax_classifier"):
        W_fc = weight_variable([size, FLAGS.num_class], "softmax_w", wd=False)
        b_fc = bias_variable([FLAGS.num_class], "softmax_b")
    Logits = tf.matmul(embedding, W_fc) + b_fc
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=Logits))
    return cross_entropy, W_fc, b_fc


def genSoftmax(embedding_anc, embedding_neg, W_fc, b_fc, label, Loss_type=FLAGS.LossType):
    if Loss_type == 'NpairLoss':
        label_split = tf.split(label, 2, axis=0)
        label_pos = tf.reshape(label_split[1], [int(FLAGS.batch_size/2), 1])
        label_neg_tile = tf.tile(label_pos, [int(FLAGS.batch_size/2), 1])

        pull_Logits = tf.matmul(embedding_neg, W_fc) + b_fc
        anc_Logits = tf.matmul(embedding_anc, W_fc) + b_fc
        label_neg_tile_2 = tf.reshape(label_neg_tile, [-1])
        label_anc_2 = tf.reshape(label_split[0], [-1])
        gen_cross_entropy = FLAGS.Softmax_factor * FLAGS._lambda * (
                tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_neg_tile_2, logits=pull_Logits))
                + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_anc_2, logits=anc_Logits))
        )
        return gen_cross_entropy
