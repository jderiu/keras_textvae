import keras.backend as K
import numpy as np
from keras.layers import Lambda, Conv1D, Conv2DTranspose, Embedding, Input, BatchNormalization, Activation, Flatten, \
    Dense, Reshape, concatenate, LSTM, ZeroPadding1D, Layer
from keras.metrics import binary_crossentropy
from keras.models import Model
from theano import tensor as T

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
    batch_size = config_data['batch_size']
    nfilter = 128
    out_size = 200
    eps = 0.001
    anneal_start = 1000.0
    anneal_end = anneal_start + 7000.0
    # == == == == == =
    # Define Encoder
    # == == == == == =
    input_idx = Input(batch_shape=(None, sample_size), dtype='float32', name='character_input')

    one_hot_weights = np.identity(nclasses)
    #oshape = (batch_size, sample_size, nclasses)
    one_hot_embeddings = Embedding(
        input_length=sample_size,
        input_dim=nclasses,
        output_dim=nclasses,
        weights=[one_hot_weights],
        trainable=False,
        name='one_hot_embeddings'
    )

    input_one_hot_embeddings = one_hot_embeddings(input_idx)
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
    hidden_zvalues = Dense(z_size*2)(hidden_intermediate_enc)
    sampling_object = Sampling(z_size)
    sampling = sampling_object(hidden_zvalues)

    # == == == == == == == =
    # Define Decoder Layers
    # == == == == == == == =
    decoder_input_layer = Dense(intermediate_dim, name='intermediate_decoding')
    hidden_intermediate_dec = decoder_input_layer(sampling)
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
    softmax_auxiliary = Dense(nclasses, activation='softmax', name='auxiliary_softmax_layer')(reshape)

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    def remove_last_column(x):
        return x[:, :-1, :]

    padding = ZeroPadding1D(padding=(1, 0))(input_one_hot_embeddings)
    previous_char_slice = Lambda(remove_last_column, output_shape=(sample_size, nclasses))(padding)

    combined_input = concatenate(inputs=[softmax_auxiliary, previous_char_slice], axis=2)
    #MUST BE IMPLEMENTATION 1 or 2
    lstm = LSTM(200, return_sequences=True, implementation=2)
    recurrent_component = lstm(combined_input)
    final_softmax_layer = Dense(nclasses, activation='softmax', name='final_softmax_layer')

    softmax_final = final_softmax_layer(recurrent_component)

    def vae_loss(args):
        x_turth, x_decoded_final, x_decoded_aux = args
        # NOTE: binary_crossentropy expects a batch_size by dim
        # for x and x_decoded_mean, so we MUST flatten these!
        x_turth = K.flatten(K.clip(x_turth, 1e-5, 1 - 1e-5))
        x_decoded_final = K.flatten(x_decoded_final)
        x_decoded_aux = K.flatten(x_decoded_aux)
        xent_loss = nclasses*sample_size*binary_crossentropy(x_turth, x_decoded_final)
        xaux_loss = nclasses*sample_size*binary_crossentropy(x_turth, x_decoded_aux)
        kl_loss = - 0.5 * K.mean(1 + sampling_object.log_sigma - K.square(sampling_object.mu) - K.exp(sampling_object.log_sigma), axis=-1)
        kld_weight = K.clip((step - anneal_start) / (anneal_end - anneal_start), 0, 1 - eps) + eps
        return xent_loss + kl_loss*kld_weight + alpha*xaux_loss

    def identity_loss(y_true, y_pred):
        return y_pred

    loss = Lambda(vae_loss, output_shape=(1,))([input_one_hot_embeddings, softmax_final, softmax_auxiliary])

    output_gen_layer = LSTMStep(lstm, final_softmax_layer, sample_size, nclasses)(softmax_auxiliary)

    train_model = Model(inputs=[input_idx], outputs=[loss])
    train_model.compile(optimizer='adam', loss=identity_loss)

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