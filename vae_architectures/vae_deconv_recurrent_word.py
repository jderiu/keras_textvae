import keras.backend as K
import numpy as np
from keras.layers import Lambda, Conv1D, Conv2DTranspose, Embedding, Input, BatchNormalization, Activation, Flatten, \
    Dense, Reshape, concatenate, LSTM, ZeroPadding1D, Layer
from keras.optimizers import Adam, Nadam
from keras.metrics import binary_crossentropy, categorical_crossentropy
from keras.models import Model
from keras.regularizers import l2
from theano import tensor as T
from os.path import join


from vae_architectures.sampling_layer import Sampling


def vae_model(config_data, vocab, step):
    z_size = config_data['z_size']
    sample_size = config_data['max_sentence_length']
    lstm_size = config_data['lstm_size']
    alpha = config_data['alpha']
    intermediate_dim = config_data['intermediate_dim']
    nfilter = 128
    out_size = 200
    eps = 0.001
    anneal_start = 1000.0
    anneal_end = anneal_start + 7000.0

    embedding_path = join(config_data['vocab_path'], 'embedding_matrix.npy')
    embedding_matrix = np.load(open(embedding_path))
    nclasses = embedding_matrix.shape[0]
    emb_dim = embedding_matrix.shape[1]

    l2_regularizer = None
    # == == == == == =
    # Define Encoder
    # == == == == == =
    input_idx = Input(batch_shape=(None, sample_size), dtype='int32', name='word_input')

    #one_hot_weights = np.identity(nclasses)
    #oshape = (batch_size, sample_size, nclasses)
    one_hot_embeddings = Embedding(
        input_length=sample_size,
        input_dim=nclasses,
        output_dim=emb_dim,
        weights=[embedding_matrix],
        trainable=False,
        name='word_embeddings'
    )

    input_one_hot_embeddings = one_hot_embeddings(input_idx)
    #oshape = (batch_size, sample_size/2, 128)
    conv1 = Conv1D(
        filters=nfilter,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer,
        activity_regularizer=l2_regularizer
    )(input_one_hot_embeddings)
    bn1 = BatchNormalization(
        beta_regularizer=l2_regularizer,
        gamma_regularizer=l2_regularizer
    )(conv1)
    relu1 = Activation(activation='relu')(bn1)
    # oshape = (batch_size, sample_size/4, 128)
    conv2 = Conv1D(
        filters=2*nfilter,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer,
        activity_regularizer=l2_regularizer
    )(relu1)
    bn2 = BatchNormalization(
        beta_regularizer=l2_regularizer,
        gamma_regularizer=l2_regularizer
    )(conv2)
    relu2 = Activation(activation='relu')(bn2)
    #oshape = (batch_size, sample_size/4*256)
    flatten = Flatten()(relu2)
    #need to store the size of the representation after the convolutions -> needed for deconv later
    hidden_intermediate_enc = Dense(
        intermediate_dim,
        activation='relu',
        name='intermediate_encoding',
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer,
        activity_regularizer=l2_regularizer
    )(flatten)
    hidden_zvalues = Dense(z_size*2)(hidden_intermediate_enc)
    sampling_object = Sampling(z_size)
    sampling = sampling_object(hidden_zvalues)

    # == == == == == == == =
    # Define Decoder Layers
    # == == == == == == == =
    decoder_input_layer = Dense(
        intermediate_dim,
        name='intermediate_decoding',
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer,
        activity_regularizer=l2_regularizer
    )
    hidden_intermediate_dec = decoder_input_layer(sampling)
    decoder_upsample = Dense(
        int(2*nfilter*sample_size/4),
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer,
        activity_regularizer=l2_regularizer
    )(hidden_intermediate_dec)
    if K.image_data_format() == 'channels_first':
        output_shape = (2*nfilter, int(sample_size/4), 1)
    else:
        output_shape = (int(sample_size/4), 1, 2*nfilter)
    reshape = Reshape(output_shape)(decoder_upsample)
    #shape = (batch_size, filters)
    deconv1 = Conv2DTranspose(
        filters=nfilter,
        kernel_size=(3, 1),
        strides=(2, 1),
        padding='same',
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer,
        activity_regularizer=l2_regularizer
    )(reshape)
    bn3 = BatchNormalization(
        beta_regularizer=l2_regularizer,
        gamma_regularizer=l2_regularizer
    )(deconv1)
    relu3 = Activation(activation='relu')(bn3)
    deconv2 = Conv2DTranspose(
        filters=out_size,
        kernel_size=(3, 1),
        strides=(2, 1),
        padding='same',
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer,
        activity_regularizer=l2_regularizer
    )(relu3)
    bn4 = BatchNormalization(
        beta_regularizer=l2_regularizer,
        gamma_regularizer=l2_regularizer
    )(deconv2)
    relu4 = Activation(activation='relu')(bn4)
    reshape = Reshape((sample_size, out_size))(relu4)
    softmax_auxiliary = Dense(
        nclasses,
        activation='softmax',
        name='auxiliary_softmax_layer',
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer,
        activity_regularizer=l2_regularizer
    )(reshape)

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    def remove_last_column(x):
        return x[:, :-1, :]

    padding = ZeroPadding1D(padding=(1, 0))(input_one_hot_embeddings)
    previous_char_slice = Lambda(remove_last_column, output_shape=(sample_size, nclasses))(padding)

    combined_input = concatenate(inputs=[softmax_auxiliary, previous_char_slice], axis=2)
    #MUST BE IMPLEMENTATION 1 or 2
    lstm = LSTM(
        200,
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
        kl_loss = - 0.5 * K.sum(1 + sampling_object.log_sigma - K.square(sampling_object.mu) - K.exp(sampling_object.log_sigma), axis=-1)
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

    main_loss = Lambda(vae_cross_ent_loss, output_shape=(1,), name='main_loss')([input_idx, softmax_final])
    kld_loss = Lambda(vae_kld_loss, output_shape=(1,), name='kld_loss')([input_one_hot_embeddings, softmax_final, softmax_auxiliary])
    aux_loss = Lambda(vae_aux_loss, output_shape=(1,), name='auxiliary_loss')([input_idx, softmax_auxiliary])

    output_gen_layer = LSTMStep(lstm, final_softmax_layer, sample_size, nclasses)(softmax_auxiliary)

    train_model = Model(inputs=[input_idx], outputs=[main_loss, kld_loss, aux_loss])

    optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004, clipvalue=1, clipnorm=10)
    train_model.compile(optimizer=optimizer, loss=identity_loss)

    test_model = Model(inputs=[input_idx], outputs=[output_gen_layer])

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

    def call(self, x):

        ndim = x.ndim
        axes = [1, 0] + list(range(2, ndim))
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
        start_word = K.zeros_like(x_shuffled[0], name='_start_word', dtype='float32')

        # output_info = [
        #     None,
        #     dict(initial=start_word, taps=[-1]),
        #     dict(initial=initial_states[0], taps=[-1]),
        #     dict(initial=initial_states[1], taps=[-1]),
        # ]

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