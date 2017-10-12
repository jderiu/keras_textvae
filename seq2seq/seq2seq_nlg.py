import numpy as np
from theano import tensor as T

import keras.backend as K
from custom_layers.sampling_layer import Sampling
from keras.layers import Lambda, Conv1D, Conv2DTranspose, Embedding, Input, BatchNormalization, Activation, Flatten, \
    Dense, Reshape
from keras.models import Model


def seq2seq(config_data, vocab, step):
    z_size = config_data['z_size']
    sample_in_size = config_data['max_input_length']
    sample_out_size = config_data['max_output_length']
    nclasses = len(vocab) + 2
    #last available index is reserved as start character
    intermediate_dim = config_data['intermediate_dim']
    nfilter = 128
    out_size = 200
    eps = 0.001
    anneal_start = 1000.0
    anneal_end = anneal_start + 7000.0
    # == == == == == =
    # Define Encoder
    # == == == == == =
    input_idx = Input(batch_shape=(None, sample_in_size), dtype='float32', name='character_input')
    output_idx = Input(batch_shape=(None, sample_out_size), dtype='int32', name='character_output')

    one_hot_weights = np.identity(nclasses)
    #oshape = (batch_size, sample_size, nclasses)
    one_hot_embeddings = Embedding(
        input_length=sample_in_size,
        input_dim=nclasses,
        output_dim=nclasses,
        weights=[one_hot_weights],
        trainable=False,
        name='one_hot_embeddings'
    )

    input_one_hot_embeddings = one_hot_embeddings((input_idx))
    #oshape = (batch_size, sample_size/2, 128)
    conv1 = Conv1D(filters=nfilter, kernel_size=3, strides=2, padding='same')(input_one_hot_embeddings)
    bn1 = BatchNormalization()(conv1)
    relu1 = Activation(activation='relu')(bn1)
    # oshape = (batch_size, sample_size/4, 128)
    conv2 = Conv1D(filters=2*nfilter, kernel_size=3, strides=2, padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = Activation(activation='relu')(bn2)
    #oshape = (batch_size, sample_size/4*256)
    flatten = Flatten()(relu2)
    #need to store the size of the representation after the convolutions -> needed for deconv later
    hidden_intermediate_enc = Dense(intermediate_dim, activation='relu', name='intermediate_encoding')(flatten)
    hidden_mean = Dense(z_size, name='mu')(hidden_intermediate_enc)
    hidden_log_sigma = Dense(z_size, name='sigma')(hidden_intermediate_enc)

    sampling_object = Sampling(z_size)
    sampling = sampling_object([hidden_mean, hidden_log_sigma])

    # == == == == == =
    # Define Decoder
    # == == == == == =
    hidden_intermediate_dec = Dense(intermediate_dim, name='intermediate_decoding')(sampling)
    decoder_upsample = Dense(int(2*nfilter*sample_out_size/4))(hidden_intermediate_dec)
    if K.image_data_format() == 'channels_first':
        output_shape = (2*nfilter, int(sample_out_size/4), 1)
    else:
        output_shape = (int(sample_out_size/4), 1, 2*nfilter)
    reshape = Reshape(output_shape)(decoder_upsample)
    #shape = (batch_size, filters)
    deconv1 = Conv2DTranspose(filters=nfilter, kernel_size=(3, 1), strides=(2, 1), padding='same')(reshape)
    bn3 = BatchNormalization()(deconv1)
    relu3 = Activation(activation='relu')(bn3)
    deconv2 = Conv2DTranspose(filters=out_size, kernel_size=(3, 1), strides=(2, 1), padding='same')(relu3)
    bn4 = BatchNormalization()(deconv2)
    relu4 = Activation(activation='relu')(bn4)
    reshape = Reshape((sample_out_size, out_size))(relu4)
    softmax = Dense(nclasses, activation='softmax')(reshape)

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    def vae_loss(args):
        x_truth, x_decoded_final = args
        x_truth_flatten = K.flatten(x_truth)
        x_decoded_flat = K.reshape(x_decoded_final, shape=(-1, K.shape(x_decoded_final)[-1]))
        cross_ent = T.nnet.categorical_crossentropy(x_decoded_flat, x_truth_flatten)
        cross_ent = K.reshape(cross_ent, shape=(-1, K.shape(x_truth)[1]))
        sum_over_sentences = K.sum(cross_ent, axis=1)
        return sum_over_sentences

    def vae_kld_loss(args):
        mu, log_sigma = args

        kl_loss = - 0.5 * K.sum(1 + log_sigma - K.square(mu) - K.exp(log_sigma), axis=-1)
        kld_weight = K.clip((step - anneal_start) / (anneal_end - anneal_start), 0, 1 - eps) + eps
        return kl_loss*kld_weight

    def identity_loss(y_true, y_pred):
        return y_pred

    loss = Lambda(vae_loss, output_shape=(1,))([output_idx, softmax])
    kld_loss = Lambda(vae_kld_loss, output_shape=(1,), name='kld_loss')([hidden_mean, hidden_log_sigma])

    argmax = Lambda(argmax_fun, output_shape=(sample_out_size,))(softmax)

    train_model = Model(inputs=[input_idx, output_idx], outputs=[loss, kld_loss])

    test_model = Model(inputs=[input_idx], outputs=[argmax])

    return train_model, test_model
