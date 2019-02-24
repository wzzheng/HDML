
from datasets import data_provider
from lib import ResNet_Model, Loss_ops, nn_Ops, Embedding_Visualization, HDML, evaluation
import copy
from tqdm import tqdm
from tensorflow.contrib import layers
from FLAGS import *


# Create the stream of datas from dataset
streams = data_provider.get_streams(FLAGS.batch_size, FLAGS.dataSet, method, crop_size=FLAGS.default_image_size)
stream_train, stream_train_eval, stream_test = streams

regularizer = layers.l2_regularizer(FLAGS.Regular_factor)
# create a saver
# check system time
_time = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
LOGDIR = FLAGS.log_save_path+FLAGS.dataSet+'/'+FLAGS.LossType+'/'+_time+'/'

if FLAGS. SaveVal:
    nn_Ops.create_path(_time)
summary_writer = tf.summary.FileWriter(LOGDIR)


def main(_):
    if not FLAGS.LossType == 'NpairLoss':
        print("LossType n-pair-loss is required")
        return 0

    # placeholders
    x_raw = tf.placeholder(tf.float32, shape=[None, FLAGS.default_image_size, FLAGS.default_image_size, 3])
    label_raw = tf.placeholder(tf.int32, shape=[None, 1])
    with tf.name_scope('istraining'):
        is_Training = tf.placeholder(tf.bool)
    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(tf.float32)

    resnet_model = ResNet_Model.Model(resnet_size=50, bottleneck=True, num_filters=64,
                                      kernel_size=7, conv_stride=2, first_pool_size=3,
                                      first_pool_stride=2, block_sizes=[3, 4, 6, 3], block_strides=[1, 2, 2, 2],
                                      resnet_version=2)
    embedding = resnet_model(x_raw, is_Training)
    with tf.variable_scope('Classifier'):
        if FLAGS.Apply_HDML:
            embedding_y_origin = embedding

        # Batch Normalization layer 1
        embedding = nn_Ops.bn_block(
            embedding, normal=FLAGS.normalize, is_Training=is_Training, name='BN1')

        # FC layer 1
        embedding_z = nn_Ops.fc_block(
            embedding, in_d=2048, out_d=FLAGS.embedding_size,
            name='fc1', is_bn=False, is_relu=False, is_Training=is_Training)

        # Embedding Visualization
        assignment, embedding_var = Embedding_Visualization.embedding_assign(
            batch_size=256, embedding=embedding_z,
            embedding_size=FLAGS.embedding_size, name='Embedding_of_fc1')

        # conventional Loss function
        with tf.name_scope('Loss'):
            # wdLoss = layers.apply_regularization(regularizer, weights_list=None)
            def exclude_batch_norm(name):
                return 'batch_normalization' not in name and 'Generator' not in name and 'Loss' not in name

            wdLoss = FLAGS.Regular_factor * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if exclude_batch_norm(v.name)]
            )
            # Get the Label
            label = tf.reduce_mean(label_raw, axis=1, keep_dims=False)
            # For some kinds of Losses, the embedding should be l2 normed
            #embedding_l = embedding_z
            J_m = Loss_ops.Loss(embedding_z, label, FLAGS.LossType) + wdLoss

    # if HNG is applied
    if FLAGS.Apply_HDML:
        with tf.name_scope('Javg'):
            Javg = tf.placeholder(tf.float32)
        with tf.name_scope('Jgen'):
            Jgen = tf.placeholder(tf.float32)

        embedding_z_quta = HDML.Pulling(FLAGS.LossType, embedding_z, Javg)

        embedding_z_concate = tf.concat([embedding_z, embedding_z_quta], axis=0)

        # Generator
        with tf.variable_scope('Generator'):

            # generator fc3
            embedding_y_concate = nn_Ops.fc_block(
                embedding_z_concate, in_d=FLAGS.embedding_size, out_d=512,
                name='generator1', is_bn=True, is_relu=True, is_Training=is_Training
            )

            # generator fc4
            embedding_y_concate = nn_Ops.fc_block(
                embedding_y_concate, in_d=512, out_d=2048,
                name='generator2', is_bn=False, is_relu=False, is_Training=is_Training
            )

            embedding_yp, embedding_yq = HDML.npairSplit(embedding_y_concate, size=2048)

        with tf.variable_scope('Classifier'):
            embedding_z_quta = nn_Ops.bn_block(
                embedding_yq, normal=FLAGS.normalize, is_Training=is_Training, name='BN1', reuse=True)

            embedding_z_quta = nn_Ops.fc_block(
                embedding_z_quta, in_d=2048, out_d=FLAGS.embedding_size,
                name='fc1', is_bn=False, is_relu=False, reuse=True, is_Training=is_Training
            )


            embedding_zq_anc = tf.slice(
                input_=embedding_z_quta, begin=[0, 0], size=[int(FLAGS.batch_size/2), int(FLAGS.embedding_size)])
            embedding_zq_negtile = tf.slice(
                    input_=embedding_z_quta, begin=[int(FLAGS.batch_size/2), 0],
                size=[int(np.square(FLAGS.batch_size/2)), int(FLAGS.embedding_size)]
                )

        with tf.name_scope('Loss'):
            J_syn = (1. - tf.exp(-FLAGS.beta / Jgen)) * Loss_ops.new_npair_loss(
                labels=label,
                embedding_anchor=embedding_zq_anc,
                embedding_positive=embedding_zq_negtile,
                equal_shape=False, reg_lambda=FLAGS.loss_l2_reg, half_batch_size=32)
            J_m = (tf.exp(-FLAGS.beta/Jgen))*J_m
            J_metric = J_m + J_syn

            cross_entropy, W_fc, b_fc = HDML.cross_entropy(embedding=embedding_y_origin, label=label, size=2048)

            embedding_yq_anc = tf.slice(
                input_=embedding_yq, begin=[0, 0], size=[int(FLAGS.batch_size / 2), 2048])
            embedding_yq_negtile = tf.slice(
                input_=embedding_yq, begin=[int(FLAGS.batch_size / 2), 0],
                size=[int(np.square(FLAGS.batch_size / 2)), 2048]
            )
            J_recon = (1 - FLAGS._lambda) * tf.reduce_sum(tf.square(embedding_yp - embedding_y_origin))
            J_soft = HDML.genSoftmax(
                embedding_anc=embedding_yq_anc, embedding_neg=embedding_yq_negtile,
                W_fc=W_fc, b_fc=b_fc, label=label
            )
            J_gen = J_recon + J_soft

    if FLAGS.Apply_HDML:
        c_train_step = nn_Ops.training(loss=J_metric, lr=lr, var_scope='Classifier')
        g_train_step = nn_Ops.training(loss=J_gen, lr=FLAGS.lr_gen, var_scope='Generator')
        s_train_step = nn_Ops.training(loss=cross_entropy, lr=FLAGS.s_lr, var_scope='Softmax_classifier')
    else:
        train_step = nn_Ops.training(loss=J_m, lr=lr)

    # initialise the session
    with tf.Session(config=config) as sess:
        # Initial all the variables with the sess
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        var = tf.global_variables()
        var_to_restore = [var for var in var if 'resnet_model' in var.name and 'Adam' not in var.name]
        saver_2 = tf.train.Saver(var_to_restore)
        saver_2.restore(sess, '/home/zwz/Desktop/CZD/cvpr_rebuttal/pretrain_model/ResNet_Pretrain/model.ckpt-225207')
        EV_batch, EV_label = Embedding_Visualization.embedding_Visual(
            LOGDIR=LOGDIR, embedding_var=embedding_var, summary_writer=summary_writer)
        
        # learning rate
        _lr = nn_Ops.learning_rate(FLAGS.init_learning_rate, decay_step=0.5, init_tar=0., cycle=10, margin=0.0005)
        
        # Restore a checkpoint
        if FLAGS.load_formalVal:
            saver.restore(sess, FLAGS.log_save_path+FLAGS.dataSet+'/'+FLAGS.LossType+'/'+FLAGS.formerTimer)
        
        # Training
        epoch_iterator = stream_train.get_epoch_iterator()

        # collectors
        J_m_loss = nn_Ops.data_collector(tag='Jm', init=1e+6)
        J_syn_loss = nn_Ops.data_collector(tag='J_syn', init=1e+6)
        J_metric_loss = nn_Ops.data_collector(tag='J_metric', init=1e+6)
        J_soft_loss = nn_Ops.data_collector(tag='J_soft', init=1e+6)
        J_recon_loss = nn_Ops.data_collector(tag='J_recon', init=1e+6)
        J_gen_loss = nn_Ops.data_collector(tag='J_gen', init=1e+6)
        cross_entropy_loss = nn_Ops.data_collector(tag='cross_entropy', init=1e+6)
        wd_Loss = nn_Ops.data_collector(tag='weight_decay', init=1e+6)
        
        step = 0
        counter = 0

        bp_epoch = FLAGS.init_batch_per_epoch
        for batch in tqdm(copy.copy(epoch_iterator)):
            # get images and labels from batch
            x_batch_data, Label_raw = nn_Ops.batch_data(batch)
            
            if not FLAGS.Apply_HDML:
                train, J_m_var, wd_Loss_var = sess.run([train_step, J_m, wdLoss],
                                                       feed_dict={x_raw: x_batch_data, label_raw: Label_raw,
                                                                  is_Training: True, lr: 7e-5})
                J_m_loss.update(var=J_m_var)
                wd_Loss.update(var=wd_Loss_var)
            
            else:
                c_train, g_train, s_train, wd_Loss_var, J_metric_var, J_m_var, \
                    J_syn_var, J_recon_var,  J_soft_var, J_gen_var, cross_en_var = sess.run(
                        [c_train_step, g_train_step, s_train_step, wdLoss,
                         J_metric, J_m, J_syn, J_recon, J_soft, J_gen, cross_entropy],
                        feed_dict={x_raw: x_batch_data, 
                                   label_raw: Label_raw,
                                   is_Training: True, lr: _lr.get_lr(), Javg: J_m_loss.read(), Jgen: J_gen_loss.read()})
                wd_Loss.update(var=wd_Loss_var)
                J_metric_loss.update(var=J_metric_var)
                J_m_loss.update(var=J_m_var)
                J_syn_loss.update(var=J_syn_var)
                J_recon_loss.update(var=J_recon_var)
                J_soft_loss.update(var=J_soft_var)
                J_gen_loss.update(var=J_gen_var)
                cross_entropy_loss.update(cross_en_var)
            step += 1
            print('learning rate %f' % _lr.get_lr())
            
            # evaluation
            if step % bp_epoch == 0:
                sess.run(assignment, feed_dict={x_raw: EV_batch, label_raw: EV_label, is_Training: False})
                nmi_tr, f1_tr, recalls_tr = evaluation.Evaluation(
                    stream_train_eval, image_mean, sess, x_raw, label_raw, is_Training, embedding_z, 98, neighbours)
                nmi_te, f1_te, recalls_te = evaluation.Evaluation(
                    stream_test, image_mean, sess, x_raw, label_raw, is_Training, embedding_z, 98, neighbours)

                # Summary
                eval_summary = tf.Summary()
                eval_summary.value.add(tag='train nmi', simple_value=nmi_tr)
                eval_summary.value.add(tag='train f1', simple_value=f1_tr)
                for i in range(0, np.shape(neighbours)[0]):
                    eval_summary.value.add(tag='Recall@%d train' % neighbours[i], simple_value=recalls_tr[i])
                eval_summary.value.add(tag='test nmi', simple_value=nmi_te)
                eval_summary.value.add(tag='test f1', simple_value=f1_te)
                for i in range(0, np.shape(neighbours)[0]):
                    eval_summary.value.add(tag='Recall@%d test' % neighbours[i], simple_value=recalls_te[i])
                J_m_loss.write_to_tfboard(eval_summary)
                wd_Loss.write_to_tfboard(eval_summary)
                if FLAGS.Apply_HDML:
                    J_syn_loss.write_to_tfboard(eval_summary)
                    J_metric_loss.write_to_tfboard(eval_summary)
                    J_soft_loss.write_to_tfboard(eval_summary)
                    J_recon_loss.write_to_tfboard(eval_summary)
                    J_gen_loss.write_to_tfboard(eval_summary)
                    cross_entropy_loss.write_to_tfboard(eval_summary)
                summary_writer.add_summary(eval_summary, step)
                print('Summary written')
                summary_writer.flush()
                print("Saved")
                save_path = saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), step)
                _lr.update(nmi_te)
                counter += 1
                if counter == 10:
                    bp_epoch = FLAGS.batch_per_epoch
                if step >= FLAGS.max_steps:
                    os._exit()


if __name__ == '__main__':
    tf.app.run()
