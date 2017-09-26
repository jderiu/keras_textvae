from theano import tensor as T

import keras.backend as K
from custom_layers.sampling_layer import Sampling
from keras.layers import Lambda, Conv1D, Conv2DTranspose, Input, BatchNormalization, Flatten, \
    Dense, Reshape, concatenate, LSTM, ZeroPadding1D, Layer, PReLU
from custom_layers.sampling_layer import Sampling
from keras.models import Model


def get_encoder(config_data, input_idx, input_one_hot_embeddings, nfilter, name, z_size):
    intermediate_dim = config_data['intermediate_dim']

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
        filters=2 * nfilter,
        kernel_size=3,
        strides=2,
        padding='same'
    )(relu1)
    bn2 = BatchNormalization(
        scale=False
    )(conv2)
    relu2 = PReLU()(bn2)
    # oshape = (batch_size, sample_size/4*256)
    flatten = Flatten()(relu2)
    # need to store the size of the representation after the convolutions -> needed for deconv later
    hidden_intermediate_enc = Dense(
        intermediate_dim,
        name='intermediate_encoding'
    )(flatten)
    hidden_zvalues = Dense(z_size * 2)(hidden_intermediate_enc)

    sampling_object = Sampling(z_size)
    sampling = sampling_object(hidden_zvalues)

    encoder = Model(inputs=input_idx, outputs=sampling, name='encoder_{}'.format(name))

    return encoder, sampling_object


def get_decoder(decoder_input, nfilter, sample_size, out_size, intermediate_dim, name):
    decoder_input_layer = Dense(intermediate_dim, name='intermediate_decoding')
    hidden_intermediate_dec = decoder_input_layer(decoder_input)
    decoder_upsample = Dense(int(2 * nfilter * sample_size / 4))(hidden_intermediate_dec)
    relu_int = PReLU()(decoder_upsample)
    if K.image_data_format() == 'channels_first':
        output_shape = (2 * nfilter, int(sample_size / 4), 1)
    else:
        output_shape = (int(sample_size / 4), 1, 2 * nfilter)
    reshape = Reshape(output_shape)(relu_int)
    # shape = (batch_size, filters)
    deconv1 = Conv2DTranspose(filters=nfilter, kernel_size=(3, 1), strides=(2, 1), padding='same')(reshape)
    bn3 = BatchNormalization(scale=False)(deconv1)
    relu3 = PReLU()(bn3)
    deconv2 = Conv2DTranspose(filters=out_size,  kernel_size=(3, 1), strides=(2, 1), padding='same')(relu3)
    bn4 = BatchNormalization(scale=False)(deconv2)
    relu4 = PReLU()(bn4)
    reshape = Reshape((sample_size, out_size))(relu4)

    decoder = Model(inputs=decoder_input, outputs=reshape, name='decoder_{}'.format(name))

    return decoder


def vae_model(config_data,input_idx_char, input_idx_word, original_sample_char, original_sample_word, vocab_char, step):
    z_size_char = 50
    sample_size = config_data['max_sentence_length']
    nchars = len(vocab_char) + 2
    #last available index is reserved as start character
    lstm_size = config_data['lstm_size']
    alpha = config_data['alpha']
    intermediate_dim = config_data['intermediate_dim']

    nfilter = 128
    out_size = 200
    eps = 0.001
    kld_anneal_start = 40000.0
    kld_anneal_end = kld_anneal_start + 80000.0

    aux_anneal_start = 0.0
    aux_anneal_end = aux_anneal_start + 20000.0

    main_anneal_start = 0.0
    main_anneal_end = main_anneal_start + 20000.0

    l2_regularizer = None

    # == == == == == =
    # Define Char VAE
    # == == == == == =
    decoder_input_char = Input(shape=(z_size_char,), name='decoder_input')
    encoder_char, sampling_obj_char = get_encoder(config_data, input_idx_char, original_sample_char, nfilter, 'char', z_size_char)
    decoder_char = get_decoder(decoder_input_char, nfilter, sample_size, out_size, intermediate_dim, 'char')

    Z_char = encoder_char(input_idx_char)
    x_char = decoder_char(Z_char)

    # == == == == == =
    # Define Word VAE
    # == == == == == =
    z_size_word = 200
    decoder_input_word = Input(shape=(z_size_word,), name='decoder_input')
    encoder_word, sampling_obj_word = get_encoder(config_data, input_idx_word, original_sample_word, nfilter, 'word', z_size_word)
    decoder_word = get_decoder(decoder_input_word, nfilter, sample_size, out_size, intermediate_dim, 'word')

    Z_word = encoder_word(input_idx_word)
    x_word = decoder_word(Z_word)

    # == == == == == =
    # Conbined VAE
    # == == == == == =

    auxiliary_char = Dense(
        nchars,
        activation='softmax',
        name='auxiliary_softmax_layer',
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer,
        activity_regularizer=l2_regularizer
    )(x_char)

    hidden = Dense(out_size, activation='linear')(x_word)
    hidden = Dense(out_size, activation='linear')(hidden)
    auxiliary_word = Dense(out_size, activation='linear', name='auxiliary_output')(hidden)

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    def remove_last_column(x):
        return x[:, :-1, :]

    padding = ZeroPadding1D(padding=(1, 0))

    padding_char = padding(original_sample_char)
    previous_char_slice = Lambda(remove_last_column, output_shape=(sample_size, nchars))(padding_char)

    combined_input = concatenate(inputs=[auxiliary_char, auxiliary_word, previous_char_slice], axis=2)
    #MUST BE IMPLEMENTATION 1 or 2
    lstm = LSTM(
        lstm_size,
        return_sequences=True,
        implementation=2
    )
    recurrent_component = lstm(combined_input)
    final_softmax_layer = Dense(
        nchars,
        activation='softmax',
        name='final_softmax_layer'
    )

    softmax_final = final_softmax_layer(recurrent_component)

    def vae_cross_ent_loss(args):
        x_truth, x_decoded_final = args
        x_truth_flatten = K.flatten(x_truth)
        x_decoded_flat = K.reshape(x_decoded_final, shape=(-1, K.shape(x_decoded_final)[-1]))
        cross_ent = T.nnet.categorical_crossentropy(x_decoded_flat, x_truth_flatten)
        cross_ent = K.reshape(cross_ent, shape=(-1, K.shape(x_truth)[1]))
        sum_over_sentences = K.sum(cross_ent, axis=1)

        loss_weight = K.clip((step - main_anneal_start) / (main_anneal_end - main_anneal_start), eps, 1 - eps)
        return sum_over_sentences*loss_weight

    def vae_kld_char_loss(args):
        kl_loss = - 0.5 * K.sum(1 + sampling_obj_char.log_sigma - K.square(sampling_obj_char.mu) - K.exp(sampling_obj_char.log_sigma), axis=-1)
        kld_weight = K.clip((step - kld_anneal_start) / (kld_anneal_end - kld_anneal_start), eps, 1 - eps)
        return kl_loss*kld_weight

    def vae_kld_word_loss(args):
        kl_loss = - 0.5 * K.sum(1 + sampling_obj_word.log_sigma - K.square(sampling_obj_word.mu) - K.exp(sampling_obj_word.log_sigma), axis=-1)
        kld_weight = K.clip((step - kld_anneal_start) / (kld_anneal_end - kld_anneal_start), eps, 1 - eps)
        return kl_loss*kld_weight

    def vae_aux_loss(args):
        x_truth, x_decoded = args
        x_truth_flatten = K.flatten(x_truth)
        x_decoded_flat = K.clip(K.reshape(x_decoded, shape=(-1, K.shape(x_decoded)[-1])), min_value=1e-5, max_value=1 - 1e-5)
        cross_ent = T.nnet.categorical_crossentropy(x_decoded_flat, x_truth_flatten)
        cross_ent = K.reshape(cross_ent, shape=(-1, K.shape(x_truth)[1]))
        sum_over_sentences = K.sum(cross_ent, axis=1)

        loss_weight = K.clip((step - aux_anneal_end) / (aux_anneal_start - aux_anneal_end), alpha, 1 - eps)
        return loss_weight*sum_over_sentences

    def vae_cosine_distance_loss(args):
        x_truth, x_decoded_final = args

        #normalize over embedding-dimension
        xt_mag = K.l2_normalize(x_truth, axis=2) #None, 40, 200
        xp_mag = K.l2_normalize(x_decoded_final, axis=2)#None, 40, 200

        elem_mult = xt_mag*xp_mag
        cosine_sim = K.sum(elem_mult, axis=2) #None, 40

        cosine_distance = 1 - cosine_sim #size = None, 40

        sum_over_sentences = K.sum(cosine_distance, axis=1) #None
        loss_weight = K.clip((step - aux_anneal_end) / (aux_anneal_start - aux_anneal_end), alpha, 1 - eps)
        return loss_weight*sum_over_sentences

    def vae_eucledian_distance_loss(args):
        x_truth, x_decoded_final = args

        squared_dist = K.square(x_truth - x_decoded_final)
        summed_dist = K.sum(squared_dist, axis=2) #None, 40
        root_dist = K.sqrt(summed_dist)
        sum_over_sentences = K.sum(root_dist, axis=1)#None
        loss_weight = K.clip((step - aux_anneal_end) / (aux_anneal_start - aux_anneal_end), alpha, 1 - eps)
        return loss_weight*sum_over_sentences

    def identity_loss(y_true, y_pred):
        return y_pred

    main_loss = Lambda(vae_cross_ent_loss, output_shape=(1,), name='main_loss')([input_idx_char, softmax_final])
    kld_loss = Lambda(vae_kld_char_loss, output_shape=(1,), name='kld_loss_char')([original_sample_char])
    kld_loss_word = Lambda(vae_kld_word_loss, output_shape=(1,), name='kld_loss_word')([original_sample_word])
    aux_loss = Lambda(vae_aux_loss, output_shape=(1,), name='auxiliary_loss')([input_idx_char, auxiliary_char])
    aux_word_loss = Lambda(vae_eucledian_distance_loss, output_shape=(1,), name='auxiliary_word_loss')([original_sample_word, auxiliary_word])

    output_gen_layer = LSTMStep(lstm, final_softmax_layer, sample_size, nchars)([auxiliary_char, auxiliary_word])

    train_model = Model(inputs=[input_idx_char, input_idx_word], outputs=[main_loss, kld_loss, kld_loss_word, aux_loss, aux_word_loss], name='vae_train_model')

    test_model = Model(inputs=[input_idx_char, input_idx_word], outputs=[output_gen_layer], name='vae_inference_model')

    final_output_model = Model(inputs=[input_idx_char, input_idx_word], outputs=[softmax_final], name='final_output_model')
    aux_char_model = Model(inputs=[input_idx_char, input_idx_word], outputs=[auxiliary_char], name='aux_char_model')
    aux_word_model = Model(inputs=[input_idx_char, input_idx_word], outputs=[auxiliary_word], name='aux_word_model')

    return train_model, test_model, aux_char_model, aux_word_model, final_output_model


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

    def call(self, inputs, **kwargs):
        x_char = inputs[0]
        x_word = inputs[1]

        x = K.concatenate(tensors=[x_char, x_word])

        ndim = x.ndim
        axes = [1, 0] + list(range(2, ndim))
        #(sample_size, batch_size, lstm_size) since we iterate over the samples
        x_shuffled = x.dimshuffle(axes)
        x_c_shuffled = x_char.dimshuffle(axes)

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
        start_char = K.ones_like(x_c_shuffled[0], name='_start_word', dtype='float32')*1e-5

        output_info = [
            None,
            dict(initial=start_char, taps=[-1]),
            dict(initial=initial_states[0], taps=[-1]),
            dict(initial=initial_states[1], taps=[-1]),
        ]

        indices = list(range(self.input_length))

        successive_words = []
        states = initial_states
        for i in indices:
            output = _step(x_shuffled[i], [start_char] + states + constants)
            start_char = output[1]
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