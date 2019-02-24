from datasets import data_provider
from FLAGS import *
import copy
from tqdm import tqdm


def create_embedding_visual_batch(logdir, batch_size=256):
    streams = data_provider.get_streams(1, FLAGS.dataSet,
                                        'clustering', crop_size=FLAGS.default_image_size)
    stream_train, stream_train_eval, stream_test = streams
    len = 0
    for batch in tqdm(copy.copy(stream_train_eval.get_epoch_iterator())):
        # get images and labels from batch
        x_batch_data, c_batch_data = batch
        # change images to [B,S,S,C]
        x_batch_data = np.transpose(x_batch_data[:,[2,1,0],:,:],(0,2,3,1))
        # Reduce mean
        x_batch_data = x_batch_data-image_mean
        Label_raw = np.reshape(c_batch_data,[1])
        if Label_raw<256:
            print(Label_raw)
            if len == 0:
                images = x_batch_data
                labels = Label_raw
            else:
                images = np.concatenate([images, x_batch_data], axis=0)
                labels = np.concatenate([labels, Label_raw], axis=0)
            len +=1
        if len >= batch_size:
            print(labels)
            with open(logdir+'metadata.tsv','w') as f:
                for i in range(batch_size):
                    c = labels[i]
                    f.write('{}\n'.format(c))
            return images, labels


def embedding_assign(batch_size, embedding, embedding_size, name):
    embedding_var = tf.Variable(tf.zeros([batch_size, embedding_size]),
                                name=name, trainable=False)
    assignment = embedding_var.assign(embedding)
    return assignment, embedding_var


def embedding_Visual(LOGDIR, embedding_var, summary_writer):
    EV_batch, EV_label = create_embedding_visual_batch(LOGDIR, 256)
    EV_label = np.reshape(EV_label, [256, 1])
    config_EV = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config_EV.embeddings.add()
    embedding_config.metadata_path = LOGDIR + 'metadata.tsv'
    embedding_config.tensor_name = embedding_var.name
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config_EV)
    return EV_batch, EV_label
