from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
import lib.nn_Ops as nn_Ops
from FLAGS import *
"""
Classic Metric Learning Losses
"""


def remove_zero(has_zero):
    return tf.reshape(tf.gather(params=has_zero, indices=tf.where(tf.not_equal(has_zero, tf.zeros(shape=tf.shape(has_zero), dtype=tf.float32)))), [-1])


def all_in(loss_vector):
    mean, var = tf.nn.moments(loss_vector, axes=0, keep_dims=False)
    return tf.logical_and(tf.not_equal(tf.shape(tf.reshape(tf.gather(params=loss_vector, indices=tf.where(
                                    tf.greater(loss_vector, (mean+3.*tf.sqrt(var))*tf.ones(tf.shape(loss_vector), dtype=tf.float32))
                                )), [-1]))[0], 0), tf.not_equal(tf.shape(tf.reshape(tf.gather(params=loss_vector, indices=tf.where(
                                    tf.greater((mean-3.*tf.sqrt(var))*tf.ones(tf.shape(loss_vector), dtype=tf.float32), loss_vector)
                                )), [-1]))[0], 0))


def remove(loss_vector):
    loss_vector = tf.reshape(tf.gather(params=loss_vector, indices=tf.where(
        tf.greater(tf.reduce_max(loss_vector) * tf.ones(tf.shape(loss_vector), dtype=tf.float32), loss_vector)
    )), [-1])
    loss_vector = tf.reshape(tf.gather(params=loss_vector, indices=tf.where(
        tf.greater(loss_vector, tf.reduce_min(loss_vector) * tf.ones(tf.shape(loss_vector), dtype=tf.float32))
    )), [-1])
    return loss_vector


def trunc_loss(loss_vector):
    loss_vector = tf.reshape(loss_vector, shape=[tf.shape(loss_vector)[0]])

    loss_vector = tf.while_loop(all_in, remove, [loss_vector])
    return loss_vector


def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
        feature: 2-D Tensor of size [number of data, feature dimension].
        squared: Boolean, whether or not to square the pairwise distances.
    Returns:
        pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(
            math_ops.square(feature),
            axis=[1],
            keep_dims=True),
        math_ops.reduce_sum(
            math_ops.square(
                array_ops.transpose(feature)),
            axis=[0],
            keep_dims=True)) - 2.0 * math_ops.matmul(
        feature, array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def contrastive_loss(labels, embeddings_anchor, embeddings_positive,
                     margin=1.0):
    """Computes the contrastive loss.
    This loss encourages the embedding to be close to each other for
        the samples of the same label and the embedding to be far apart at least
        by the margin constant for the samples of different labels.
    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
            binary labels indicating positive vs negative pair.
        embeddings_anchor: 2-D float `Tensor` of embedding vectors for the anchor
            images. Embeddings should be l2 normalized.
        embeddings_positive: 2-D float `Tensor` of embedding vectors for the
            positive images. Embeddings should be l2 normalized.
        margin: margin term in the loss definition.
    Returns:
        contrastive_loss: tf.float32 scalar.
    """
    # Get per pair distances
    distances = math_ops.sqrt(
        math_ops.reduce_sum(
            math_ops.square(embeddings_anchor - embeddings_positive), 1))

    # Add contrastive loss for the siamese network.
    #   label here is {0,1} for neg, pos.
    return math_ops.reduce_mean(
        math_ops.to_float(labels) * math_ops.square(distances) +
        (1. - math_ops.to_float(labels)) *
        math_ops.square(math_ops.maximum(margin - distances, 0.)),
        name='contrastive_loss')


def contrastive_loss_v2(pos1, pos2, neg1, neg2, alpha=1.0):
    distance = tf.reduce_sum((pos1 - pos2) ** 2.0, axis=1) + \
               tf.nn.relu(
                   alpha - tf.reduce_sum((neg1 - neg2) ** 2.0, axis=1))
    return tf.reduce_mean(distance) / 4


def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
      Args:
        data: 2-D float `Tensor` of size [n, m].
        mask: 2-D Boolean `Tensor` of size [n, m].
        dim: The dimension over which to compute the maximum.
      Returns:
        masked_maximums: N-D `Tensor`.
          The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keep_dims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(
            data - axis_minimums, mask), dim, keep_dims=True) + axis_minimums
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.
  Returns:
    masked_minimums: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
  """
    axis_maximums = math_ops.reduce_max(data, dim, keep_dims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(
            data - axis_maximums, mask), dim, keep_dims=True) + axis_maximums
    return masked_minimums


def triplet_loss(anchor, positive, negative, margin=1.0):
    d_ap = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    d_an = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(d_ap - d_an + margin, 0.)
    return tf.reduce_sum(loss)


def triplet_semihard_loss(labels, embeddings, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
      The loss encourages the positive distances (between a pair of embeddings with
      the same labels) to be smaller than the minimum negative distance among
      which are at least greater than the positive distance plus the margin constant
      (called semi-hard negative) in the mini-batch. If no such negative exists,
      uses the largest negative distance instead.
      See: https://arxiv.org/abs/1503.03832.
      Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
          multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
          be l2 normalized.
        margin: Float, margin term in the loss definition.
      Returns:
        triplet_loss: tf.float32 scalar.
    """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    batch_size = array_ops.size(labels)

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(
                    mask, dtype=dtypes.float32), 1, keep_dims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    _triplet_loss = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')
    return _triplet_loss


def new_npair_loss(labels, embedding_anchor, embedding_positive, reg_lambda, equal_shape=True, half_batch_size=64):
    """
    N-pair loss used in HDML, the inputs are constructed into N/2 N/2+1 pairs
    :param labels: A 1-d tensor of size [batch_size], which presents the sparse label of the embedding
    :param embedding_anchor: A 4-d tensor of size [batch_size/2, H, W, C], the embedding of anchor
    :param embedding_positive: A 4-d tensor of size [batch_size/2, H, W, C], the embedding of positive
    :param reg_lambda: float, the l2-regular factor of N-pair Loss
    :param equal_shape: boolean, whether shape(embedding_anchor)[0] == shape(embedding_positive)[0]
    :param half_batch_size: int, if batch size == 128, half_batch_size will be 64
    :return: The n-pair loss, which equals to npair_loss + reg_lambda*l2_loss
    """
    reg_anchor = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embedding_anchor), 1))
    reg_positive = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embedding_positive), 1))
    l2loss = math_ops.multiply(
        0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')
    xent_loss = []
    if equal_shape:
        pos_tile = tf.tile(embedding_positive, [half_batch_size, 1], name='pos_tile')
    else:
        pos_tile = embedding_positive
    anc = tf.split(embedding_anchor, half_batch_size, axis=0)
    pos = tf.split(pos_tile, half_batch_size, axis=0)
    label2 = tf.split(labels, 2, axis=0)
    label_anc = tf.reshape(label2[0], [half_batch_size, 1])
    label_pos = tf.reshape(label2[1], [half_batch_size, 1])
    label_anc = tf.split(label_anc, half_batch_size, axis=0)

    for i in range(half_batch_size):
        similarity_matrix = tf.matmul(anc[i], pos[i], transpose_a=False, transpose_b=True)
        anc_label = tf.reshape(label_anc[i], [1, 1])
        pos_label = tf.reshape(label_pos, [half_batch_size, 1])
        labels_remapped = tf.to_float(
            tf.equal(anc_label, tf.transpose(pos_label))
        )
        labels_remapped /= tf.reduce_sum(labels_remapped, 1, keep_dims=True)

        x_loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=similarity_matrix, labels=labels_remapped
        )
        xent_loss.append(x_loss)

    xent_loss = tf.reduce_mean(xent_loss, name='xentrop')
    r_loss = tf.cond(tf.is_nan(xent_loss + l2loss), lambda: tf.constant(0.), lambda: xent_loss + l2loss)
    return r_loss


def npairs_loss(labels, embeddings_anchor, embeddings_positive,
                reg_lambda=3e-3, print_losses=False):
    """Computes the npairs loss.
          Npairs loss expects paired data where a pair is composed of samples from the
          same labels and each pairs in the minibatch have different labels. The loss
          has two components. The first component is the L2 regularizer on the
          embedding vectors. The second component is the sum of cross entropy loss
          which takes each row of the pair-wise similarity matrix as logits and
          the remapped one-hot labels as labels.
          See:
          http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
          Args:
            labels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
            embeddings_anchor: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
              embedding vectors for the anchor images. Embeddings should not be
              l2 normalized.
            embeddings_positive: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
              embedding vectors for the positive images. Embeddings should not be
              l2 normalized.
            reg_lambda: Float. L2 regularization term on the embedding vectors.
            print_losses: Boolean. Option to print the xent and l2loss.
          Returns:
            npairs_loss: tf.float32 scalar.
      """
    # pylint: enable=line-too-long
    # Add the regularizer on the embedding.
    reg_anchor = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
    reg_positive = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
    l2loss = math_ops.multiply(
        0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

    # Get per pair similarities.
    similarity_matrix = math_ops.matmul(
        embeddings_anchor, embeddings_positive, transpose_a=False,
        transpose_b=True)

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    labels_remapped = math_ops.to_float(
        math_ops.equal(labels, array_ops.transpose(labels)))
    labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)

    # Add the softmax loss.
    xent_loss = nn.softmax_cross_entropy_with_logits(
        logits=similarity_matrix, labels=labels_remapped)
    xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')

    if print_losses:
        xent_loss = logging_ops.Print(
            xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])

    return l2loss + xent_loss


def lifted_struct_loss(labels, embeddings, margin=1.0):
    """Computes the lifted structured loss.
      The loss encourages the positive distances (between a pair of embeddings
      with the same labels) to be smaller than any negative distances (between a
      pair of embeddings with different labels) in the mini-batch in a way
      that is differentiable with respect to the embedding vectors.
      See: https://arxiv.org/abs/1511.06452.
      Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
          multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should not
          be l2 normalized.
        margin: Float, margin term in the loss definition.
      Returns:
        lifted_loss: tf.float32 scalar.
    """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pairwise_distances = pairwise_distance(embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    batch_size = array_ops.size(labels)

    diff = margin - pairwise_distances
    mask = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    # Safe maximum: Temporarily shift negative distances
    #   above zero before taking max.
    #     this is to take the max only among negatives.
    row_minimums = math_ops.reduce_min(diff, 1, keep_dims=True)
    row_negative_maximums = math_ops.reduce_max(
        math_ops.multiply(
            diff - row_minimums, mask), 1, keep_dims=True) + row_minimums

    max_elements = math_ops.maximum(
        row_negative_maximums, array_ops.transpose(row_negative_maximums))
    diff_tiled = array_ops.tile(diff, [batch_size, 1])
    mask_tiled = array_ops.tile(mask, [batch_size, 1])
    max_elements_vect = array_ops.reshape(
        array_ops.transpose(max_elements), [-1, 1])

    loss_exp_left = array_ops.reshape(
        math_ops.reduce_sum(math_ops.multiply(
            math_ops.exp(
                diff_tiled - max_elements_vect),
            mask_tiled), 1, keep_dims=True), [batch_size, batch_size])

    loss_mat = max_elements + math_ops.log(
        loss_exp_left + array_ops.transpose(loss_exp_left))
    # Add the positive distance.
    loss_mat += pairwise_distances

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
    num_positives = math_ops.reduce_sum(mask_positives) / 2.0

    lifted_loss = math_ops.truediv(
        0.25 * math_ops.reduce_sum(
            math_ops.square(
                math_ops.maximum(
                    math_ops.multiply(loss_mat, mask_positives), 0.0))),
        num_positives,
        name='liftedstruct_loss')
    return lifted_loss


def Binary_sofmax(anc_emb, pos_emb, label, cycle, weight, emb_size):
    cross_entropy = tf.constant(0., tf.float32)
    with tf.variable_scope("Softmax_classifier"):
        W_fc = nn_Ops.weight_variable([2048, 2], "softmax_w", wd=False)
        b_fc = nn_Ops.bias_variable([2], "softmax_b")
    for i in range(cycle):
        if i >= 64:
            break
        pos_f = tf.slice(input_=pos_emb, begin=[0, 0], size=[i, emb_size])
        label_f = tf.slice(input_=label, begin=[0], size=[i])
        pos_b = tf.slice(input_=pos_emb, begin=[i, 0], size=[64 - i, emb_size])
        label_b = tf.slice(input_=label, begin=[i], size=[64 - i])
        pos_temp = tf.concat([pos_b, pos_f], axis=0)
        label_temp = tf.concat([label_b, label_f], axis=0)
        logits = tf.matmul(tf.concat([anc_emb, pos_temp], axis=1), W_fc) + b_fc
        label_binary = tf.cast(tf.equal(label, label_temp), tf.int32)
        weight_m = tf.cast(tf.logical_not(tf.equal(label, label_temp)), tf.float32) \
                   * weight + tf.cast(label_binary, tf.float32)
        cross_entropy += tf.reduce_mean(
            tf.multiply(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                               labels=label_binary), weight_m))/np.float32(cycle)
    return cross_entropy, W_fc, b_fc


def Loss(embedding, label, _lossType="Softmax", loss_l2_reg=FLAGS.loss_l2_reg):
    embedding_split = tf.split(embedding, 2, axis=0)
    label_split = tf.split(label, 2, axis=0)
    embedding_anchor = embedding_split[0]
    embedding_positive = embedding_split[1]
    label_positive = label_split[1]
    _Loss = 0

    if _lossType == "Softmax":
        print("Use Softmax")
        W_fc2 = nn_Ops.weight_variable([1024, 10])
        b_fc2 = nn_Ops.bias_variable([10])
        y_conv = tf.matmul(embedding, W_fc2) + b_fc2
        _Loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y_conv))

    elif _lossType == "Contrastive_Loss":
        print("Use Contrastive_Loss_v2")
        embedding_anchor = tf.split(embedding_anchor, 2, axis=0)
        embedding_positive = tf.split(embedding_positive, 2, axis=0)
        _Loss = contrastive_loss_v2(
            embedding_anchor[0], embedding_anchor[1], embedding_positive[0], embedding_positive[1], alpha=1.0)

    elif _lossType == "Triplet_Semihard":
        print("Use Triplet_semihard")
        _Loss = triplet_semihard_loss(label, embedding)

    elif _lossType == "LiftedStructLoss":
        print("Use LiftedStructLoss")
        _Loss = lifted_struct_loss(label, embedding)

    elif _lossType == "NpairLoss":
        print("Use NpairLoss")
        _Loss = npairs_loss(label_positive, embedding_anchor, embedding_positive, reg_lambda=loss_l2_reg)

    elif _lossType == "Triplet":
        print("Use Triplet Loss")
        embedding3 = tf.split(embedding, 3, axis=0)
        anchor = embedding3[0]
        positive = embedding3[1]
        negative = embedding3[2]
        _Loss = triplet_loss(anchor, positive, negative)
        
    elif _lossType == "New_npairLoss":
        print("Use new NpairLoss")
        _Loss = new_npair_loss(
            labels=label, embedding_anchor=embedding_anchor,
            embedding_positive=embedding_positive, reg_lambda=loss_l2_reg,
            equal_shape=True, half_batch_size=int(FLAGS.batch_size/2))

    return _Loss


