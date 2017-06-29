from keras.layers import Lambda, Conv1D, Conv2DTranspose, Embedding, Input, BatchNormalization, Activation, Flatten, Dense, Reshape, TimeDistributed
from sampling_layer import Sampling
from keras.models import Model
import keras.backend as K
from keras.metrics import binary_crossentropy, kullback_leibler_divergence
import numpy as np


def vae_model(config_data, vocab):
    z_size = config_data['z_size']
    sample_size = config_data['max_sentence_length']
    nclasses = len(vocab) + 2
    lstm_size = config_data['lstm_size']
    alpha = config_data['alpha']
    intermediate_dim = config_data['intermediate_dim']
    nfilter = 128
    out_size = 200


    # == == == == == =
    # Define Encoder
    # == == == == == =
    input_idx = Input(batch_shape=(None, sample_size, nclasses), dtype='float32', name='character_input')
    # one_hot_weights = np.identity(nclasses)
    # #oshape = (batch_size, sample_size, nclasses)
    # one_hot_embeddings = Embedding(
    #     input_length=sample_size,
    #     input_dim=nclasses,
    #     output_dim=nclasses,
    #     weights=[one_hot_weights],
    #     trainable=False,
    #     name='one_hot_embeddings'
    # )(input_idx)
    #oshape = (batch_size, sample_size/2, 128)
    conv1 = Conv1D(filters=nfilter, kernel_size=3, strides=2, padding='same')(input_idx)
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
    hidden_zvalues = Dense(z_size*2)(hidden_intermediate_enc)
    sampling_object = Sampling(z_size)
    sampling = sampling_object(hidden_zvalues)

    # == == == == == =
    # Define Decoder
    # == == == == == =
    hidden_intermediate_dec = Dense(intermediate_dim, name='intermediate_decoding')(sampling)
    decoder_upsample = Dense(int(2*nfilter*sample_size/4))(hidden_intermediate_dec)
    if K.image_data_format() == 'channels_first':
        output_shape = (2*nfilter, int(sample_size/4), 1)
    else:
        output_shape = (int(sample_size/4), 1, 2*nfilter)
    reshape = Reshape(output_shape)(decoder_upsample)
    #shape = (batch_size, filters)
    deconv1 = Conv2DTranspose(filters=nfilter, kernel_size=(3, 1), strides=(2, 1), padding='same')(reshape)
    bn3 = BatchNormalization()(deconv1)
    relu3 = Activation(activation='relu')(bn3)
    deconv2 = Conv2DTranspose(filters=out_size, kernel_size=(3, 1), strides=(2, 1), padding='same')(relu3)
    bn4 = BatchNormalization()(deconv2)
    relu4 = Activation(activation='relu')(bn4)
    reshape = Reshape((sample_size, out_size))(relu4)
    softmax = Dense(nclasses, activation='softmax')(reshape)

    def vae_loss(x, x_decoded_mean):
        # NOTE: binary_crossentropy expects a batch_size by dim
        # for x and x_decoded_mean, so we MUST flatten these!
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + sampling_object.log_sigma - K.square(sampling_object.mu) - K.exp(sampling_object.log_sigma), axis=-1)
        return xent_loss + kl_loss

    def identity_loss(y_true, y_pred):
        return y_pred

    #loss = Lambda(vae_loss, output_shape=(1,))([one_hot_embeddings, softmax])

    model = Model(inputs=[input_idx], outputs=[softmax])
    model.compile(optimizer='adam', loss=vae_loss)

    return model
