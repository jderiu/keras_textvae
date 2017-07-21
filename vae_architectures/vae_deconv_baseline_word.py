import keras.backend as K
import numpy as np
from keras.layers import Lambda, Conv1D, Conv2DTranspose, Embedding, Input, BatchNormalization, Activation, Flatten, \
    Dense, Reshape, Layer
from keras.metrics import binary_crossentropy
from keras.models import Model
from os.path import join
import theano

from vae_architectures.sampling_layer import Sampling


def vae_model(config_data, vocab, step):
    z_size = config_data['z_size']
    sample_size = config_data['max_sentence_length']
    nclasses = len(vocab) + 2
    #last available index is reserved as start character
    start_word_idx = nclasses - 1
    lstm_size = config_data['lstm_size']
    alpha = config_data['alpha']
    intermediate_dim = config_data['intermediate_dim']
    nfilter = 128
    out_size = 200
    eps = 0.001
    anneal_start = 0.0
    anneal_end= anneal_start + 7000.0

    embedding_path = join(config_data['vocab_path'], 'embedding_matrix.npy')
    embedding_matrix = np.load(open(embedding_path, 'rb'))
    nclasses = embedding_matrix.shape[0]
    emb_dim = embedding_matrix.shape[1]
    # == == == == == =
    # Define Encoder
    # == == == == == =
    input_idx = Input(batch_shape=(None, sample_size), dtype='int32', name='character_input')

    #one_hot_weights = np.identity(nclasses)
    #oshape = (batch_size, sample_size, nclasses)
    word_embedding_layer = Embedding(
        input_length=sample_size,
        input_dim=nclasses,
        output_dim=emb_dim,
        weights=[embedding_matrix],
        trainable=False,
        name='word_embeddings'
    )

    input_word_embeddings = word_embedding_layer((input_idx))
    #oshape = (batch_size, sample_size/2, 128)
    conv1 = Conv1D(filters=nfilter, kernel_size=3, strides=2, padding='same')(input_word_embeddings)
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
    hidden = Dense(out_size, activation='linear')(reshape)
    hidden = Dense(out_size, activation='linear')(hidden)
    hidden = Dense(out_size, activation='linear')(hidden)


    def vae_cosine_distance_loss(args):
        x_truth, x_decoded_final = args

        #normalize over embedding-dimension
        xt_mag = K.l2_normalize(x_truth, axis=2) #None, 40, 200
        xp_mag = K.l2_normalize(x_decoded_final, axis=2)#None, 40, 200

        elem_mult = xt_mag*xp_mag
        cosine_sim = K.sum(elem_mult, axis=2) #None, 40

        cosine_distance = 1 - cosine_sim #size = None, 40

        sum_over_sentences = K.sum(cosine_distance, axis=1)#None
        return sum_over_sentences

    def vae_mse_loss(args):
        x_truth, x_decoded_final = args

        difference = x_truth - x_decoded_final
        squared_difference = K.square(difference)
        sums = K.sum(K.sum(squared_difference, axis=2), axis=1)
        return sums

    def vae_kld_loss(args):
        kl_loss = - 0.5 * K.sum(1 + sampling_object.log_sigma - K.square(sampling_object.mu) - K.exp(sampling_object.log_sigma), axis=-1)
        kld_weight = K.clip((step - anneal_start) / (anneal_end - anneal_start), 0, 1 - eps) + eps
        return kl_loss*kld_weight

    main_loss = Lambda(vae_cosine_distance_loss, output_shape=(1,), name='main_loss')([input_word_embeddings, hidden])
    kld_loss = Lambda(vae_kld_loss, output_shape=(1,), name='kld_loss')([input_word_embeddings])

    prediction = PredictionLayer(word_embedding_layer, sample_size, nclasses)(hidden)

    train_model = Model(inputs=[input_idx], outputs=[main_loss, kld_loss])

    test_model = Model(inputs=[input_idx], outputs=[prediction])

    return train_model, test_model


class PredictionLayer(Layer):
    def __init__(self, embedding_layer, input_length, nclasses, **kwargs):
        assert isinstance(embedding_layer, Embedding)
        self.c = None
        self.h = None
        self.input_length = input_length
        self.embedding_weights = K.transpose(K.l2_normalize(embedding_layer.embeddings, axis=1))
        self.nclasses = nclasses
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PredictionLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]

    def call(self, x):
        x_norm = K.l2_normalize(x, axis=2)

        #s = K.dot(x_norm, K.transpose(self.embedding_weights))
        #outputs = K.argmax(s, axis=2)

        def _step(x, e_weights):
            #x in (128, 200)
            #e in (200, 500k)
            s = K.dot(x, e_weights) #shape= 128, 500k
            idx = K.argmax(s, axis=1) #shape 128,
            #idx = K.max(s, axis=1) #shape 128,
            return [idx]

        results, _ = theano.scan(
            _step,
            sequences=x_norm,
            outputs_info=None,
            non_sequences=self.embedding_weights)

        # deal with Theano API inconsistency
        if isinstance(results, list):
            outputs = results[0]
        else:
            outputs = results

        return outputs