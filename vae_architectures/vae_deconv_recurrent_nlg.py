import keras.backend as K
import numpy as np
from keras.layers import Lambda, Conv1D, Conv2DTranspose, Embedding, Input, BatchNormalization, Activation, Flatten, \
    Dense, Reshape, concatenate, LSTM, ZeroPadding1D, Layer, PReLU

from keras.metrics import binary_crossentropy, categorical_crossentropy
from keras.models import Model
from keras.regularizers import l2
from theano import tensor as T


from vae_architectures.sampling_layer import Sampling


def get_conv_stack(input_one_hot_embeddings, nfilter):
    # oshape = (batch_size, sample_size/2, 128)
    conv1 = Conv1D(
        filters=nfilter,
        kernel_size=3,
        strides=2,
        padding='same'
    )(input_one_hot_embeddings)
    bn1 = BatchNormalization(
        scale=False
    )(conv1)
    relu1 = PReLU()(bn1)
    # oshape = (batch_size, sample_size/4, 128)
    conv2 = Conv1D(
        filters= nfilter,
        kernel_size=3,
        strides=2,
        padding='same'
    )(relu1)
    bn2 = BatchNormalization(
        scale=False
    )(conv2)
    relu2 = PReLU()(bn2)
    conv3 = Conv1D(
        filters= nfilter,
        kernel_size=3,
        strides=2,
        padding='same',
    )(relu2)
    bn3 = BatchNormalization(
        scale=False
    )(conv3)
    relu3 = PReLU()(bn3)
    # oshape = (batch_size, sample_size/4*256)
    flatten = Flatten()(relu3)

    return flatten


def get_encoder(inputs, name_one_hot_embeddings, near_one_hot_embeddings, nfilter, z_size, intermediate_dim):
    name_idx, eat_type_idx, price_range_idx, customer_feedback_idx, near_idx, food_idx, area_idx, family_idx, _ = inputs

    name_conv = get_conv_stack(name_one_hot_embeddings, nfilter)
    near_conv = get_conv_stack(near_one_hot_embeddings, nfilter)

    full_concat = concatenate(inputs=[name_conv, near_conv, eat_type_idx, price_range_idx, customer_feedback_idx, food_idx, area_idx, family_idx])

    # need to store the size of the representation after the convolutions -> needed for deconv later
    hidden_intermediate_enc = Dense(
        intermediate_dim,
        name='intermediate_encoding'
    )(full_concat)
    hidden_mean = Dense(z_size, name='mu')(hidden_intermediate_enc)
    hidden_log_sigma = Dense(z_size, name='sigma')(hidden_intermediate_enc)

    sampling_object = Sampling(z_size)
    sampling = sampling_object([hidden_mean, hidden_log_sigma])

    encoder = Model(inputs=inputs[:-1], outputs=[sampling, hidden_mean, hidden_log_sigma])

    encoder.summary()
    return encoder, [hidden_mean, hidden_log_sigma]


def get_decoder(decoder_input, intermediate_dim, nfilter,sample_out_size, out_size, nclasses):
    decoder_input_layer = Dense(
        intermediate_dim,
        name='intermediate_decoding'
    )
    hidden_intermediate_dec = decoder_input_layer(decoder_input)
    decoder_upsample = Dense(
        int(2 * nfilter * sample_out_size / 8)
    )(hidden_intermediate_dec)
    relu_int = PReLU()(decoder_upsample)
    if K.image_data_format() == 'channels_first':
        output_shape = (2 * nfilter, int(sample_out_size / 8), 1)
    else:
        output_shape = (int(sample_out_size / 8), 1, 2 * nfilter)
    reshape = Reshape(output_shape)(relu_int)
    # shape = (batch_size, filters)
    deconv1 = Conv2DTranspose(
        filters=nfilter,
        kernel_size=(3, 1),
        strides=(2, 1),
        padding='same'
    )(reshape)
    bn4 = BatchNormalization(
        scale=False
    )(deconv1)
    relu4 = PReLU()(bn4)
    deconv2 = Conv2DTranspose(
        filters=out_size,
        kernel_size=(3, 1),
        strides=(2, 1),
        padding='same'
    )(relu4)
    bn5 = BatchNormalization(
        scale=False
    )(deconv2)
    relu5 = PReLU()(bn5)
    deconv3 = Conv2DTranspose(
        filters=out_size,
        kernel_size=(3, 1),
        strides=(2, 1),
        padding='same'
    )(relu5)
    bn6 = BatchNormalization(
        scale=False
    )(deconv3)
    relu6 = PReLU()(bn6)
    reshape = Reshape((sample_out_size, out_size))(relu6)
    softmax_auxiliary = Dense(
        nclasses,
        activation='softmax',
        name='auxiliary_softmax_layer'
    )(reshape)

    decoder = Model(inputs=decoder_input, outputs=softmax_auxiliary)

    return decoder


def vae_model(config_data, vocab, step):
    z_size = config_data['z_size']
    sample_in_size = config_data['max_input_length']
    sample_out_size = config_data['max_output_length']
    nclasses = len(vocab) + 2
    #last available index is reserved as start character
    start_word_idx = nclasses - 1
    lstm_size = config_data['lstm_size']
    alpha = config_data['alpha']
    intermediate_dim = config_data['intermediate_dim']
    batch_size = config_data['batch_size']
    nfilter = 64
    out_size = 200
    eps = 0.001
    anneal_start = 0
    anneal_end = anneal_start + 10000.0

    l2_regularizer = None
    # == == == == == =
    # Define Encoder
    # == == == == == =
    name_idx = Input(batch_shape=(None, sample_in_size), dtype='int32', name='name_idx')
    eat_type_idx = Input(batch_shape=(None, 3), dtype='float32', name='eat_type_idx')
    price_range_idx = Input(batch_shape=(None, 6), dtype='float32', name='price_range_idx')
    customer_feedback_idx = Input(batch_shape=(None, 6), dtype='float32', name='customer_feedback_idx')
    near_idx = Input(batch_shape=(None, sample_in_size), dtype='float32', name='near_idx')
    food_idx = Input(batch_shape=(None, 7), dtype='float32', name='food_idx')
    area_idx = Input(batch_shape=(None, 2), dtype='float32', name='area_idx')
    family_idx = Input(batch_shape=(None, 2), dtype='float32', name='family_idx')
    output_idx = Input(batch_shape=(None, sample_out_size), dtype='int32', name='character_output')

    inputs = [name_idx, eat_type_idx, price_range_idx, customer_feedback_idx, near_idx, food_idx, area_idx, family_idx, output_idx]

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

    one_hot_out_embeddings = Embedding(
        input_length=sample_out_size,
        input_dim=nclasses,
        output_dim=nclasses,
        weights=[one_hot_weights],
        trainable=False,
        name='one_hot_out_embeddings'
    )

    name_one_hot_embeddings = one_hot_embeddings(name_idx)
    near_one_hot_embeddings = one_hot_embeddings(near_idx)

    output_one_hot_embeddings = one_hot_out_embeddings(output_idx)

    decoder_input = Input(shape=(z_size,), name='decoder_input')
    encoder, sampling_input = get_encoder(inputs, name_one_hot_embeddings, near_one_hot_embeddings, nfilter, z_size, intermediate_dim)
    decoder = get_decoder(decoder_input, intermediate_dim, nfilter, sample_out_size, out_size, nclasses)

    x_sampled, x_mean, x_los_sigma = encoder(inputs[:-1])
    softmax_auxiliary = decoder(x_sampled)

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    def remove_last_column(x):
        return x[:, :-1, :]

    padding = ZeroPadding1D(padding=(1, 0))(output_one_hot_embeddings)
    previous_char_slice = Lambda(remove_last_column, output_shape=(sample_out_size, nclasses))(padding)

    combined_input = concatenate(inputs=[softmax_auxiliary, previous_char_slice], axis=2)
    #MUST BE IMPLEMENTATION 1 or 2
    lstm = LSTM(
        lstm_size,
        return_sequences=True,
        implementation=2,
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer,
        recurrent_regularizer=l2_regularizer,
        activity_regularizer=l2_regularizer
    )
    recurrent_component = lstm(combined_input)
    final_softmax_layer = Dense(
        nclasses,
        activation='softmax',
        name='final_softmax_layer',
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer,
        activity_regularizer=l2_regularizer)

    softmax_final = final_softmax_layer(recurrent_component)

    def vae_cross_ent_loss(args):
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

    def vae_aux_loss(args):
        x_truth, x_decoded = args
        x_truth_flatten = K.flatten(x_truth)
        x_decoded_flat = K.reshape(x_decoded, shape=(-1, K.shape(x_decoded)[-1]))
        cross_ent = T.nnet.categorical_crossentropy(x_decoded_flat, x_truth_flatten)
        cross_ent = K.reshape(cross_ent, shape=(-1, K.shape(x_truth)[1]))
        sum_over_sentences = K.sum(cross_ent, axis=1)
        return alpha*sum_over_sentences

    def identity_loss(y_true, y_pred):
        return y_pred

    main_loss = Lambda(vae_cross_ent_loss, output_shape=(1,), name='main_loss')([output_idx, softmax_final])
    kld_loss = Lambda(vae_kld_loss, output_shape=(1,), name='kld_loss')([x_mean, x_los_sigma])
    aux_loss = Lambda(vae_aux_loss, output_shape=(1,), name='auxiliary_loss')([output_idx, softmax_auxiliary])

    output_gen_layer = LSTMStep(lstm, final_softmax_layer, sample_out_size, nclasses)(softmax_auxiliary)

    train_model = Model(inputs=inputs, outputs=[main_loss, kld_loss, aux_loss])
    test_model = Model(inputs=inputs, outputs=[output_gen_layer])

    return train_model, test_model


class LSTMStep(Layer):

    def __init__(self, lstm, softmax_layer, input_length, nclasses, **kwargs):
        assert isinstance(lstm, LSTM)
        assert isinstance(softmax_layer, Dense)
        self.lstm = lstm
        self.c = None
        self.h = None
        self.input_length = input_length
        self.softmax_layer = softmax_layer
        self.nclasses = nclasses
        super(LSTMStep, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LSTMStep, self).build(input_shape)

    def reset_state(self):
        self.c = None
        self.h = None

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.lstm.units])  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.lstm.states))]
        return initial_states

    def call(self, x, **kwargs):

        ndim = x.ndim
        axes = [1, 0] + list(range(2, ndim))
        #(sample_size, batch_size, lstm_size) since we iterate over the samples
        x_shuffled = x.dimshuffle(axes)

        def _step(x_i, states):
            current_word_vec = states[0]
            ins = K.concatenate(tensors=[x_i, current_word_vec], axis=1)

            output, new_states = self.lstm.step(ins, states[1:])

            outsoftmax = self.softmax_layer.call(output)
            # argmax that returns the predicted char
            word_idx = T.argmax(outsoftmax, axis=1)
            current_word_vec = K.one_hot(word_idx, self.nclasses)

            return [word_idx] + [current_word_vec] + new_states

        initial_states = self.get_initial_states(x)
        # if len(initial_states) > 0:
        #     initial_states[0] = T.unbroadcast(initial_states[0], 1)

        constants = self.lstm.get_constants(x)
        start_word = K.ones_like(x_shuffled[0], name='_start_word', dtype='float32')*1e-5

        output_info = [
            None,
            dict(initial=start_word, taps=[-1]),
            dict(initial=initial_states[0], taps=[-1]),
            dict(initial=initial_states[1], taps=[-1]),
        ]

        indices = list(range(self.input_length))

        successive_words = []
        states = initial_states
        for i in indices:
            output = _step(x_shuffled[i], [start_word] + states + constants)
            start_word = output[1]
            states = output[2:]
            successive_words.append(output[0])

        outputs = T.stack(*successive_words).dimshuffle([1, 0])

        # results, _ = theano.scan(
        #     _step,
        #     sequences=x,
        #     outputs_info=output_info,
        #     non_sequences=constants,
        #     go_backwards=self.lstm.go_backwards)

        # deal with Theano API inconsistency
        # if isinstance(results, list):
        #     outputs = results[0]
        #     states = results[1:]
        # else:
        #     outputs = results
        #     states = []

        return outputs