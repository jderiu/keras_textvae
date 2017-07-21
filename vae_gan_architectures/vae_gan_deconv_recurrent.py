import keras.backend as K
import numpy as np
from keras.layers import Lambda, Conv1D, Conv2DTranspose, Embedding, Input, BatchNormalization, Flatten, \
    Dense, Reshape, concatenate, LSTM, ZeroPadding1D, Layer, PReLU, GaussianNoise

from keras.models import Model
from keras.optimizers import RMSprop, Adadelta
import theano.tensor as T

from vae_architectures.sampling_layer import Sampling


def get_encoder(input_idx, input_one_hot_embeddings, nfilter, intermediate_dim, z_size):
    # oshape = (batch_size, sample_size/2, 128)
    conv1 = Conv1D(filters=nfilter, kernel_size=3, strides=2, padding='same')(input_one_hot_embeddings)
    bn1 = BatchNormalization(scale=False)(conv1)
    relu1 = PReLU()(bn1)
    # oshape = (batch_size, sample_size/4, 128)
    conv2 = Conv1D(filters=2 * nfilter, kernel_size=3, strides=2, padding='same')(relu1)
    bn2 = BatchNormalization(scale=False)(conv2)
    relu2 = PReLU()(bn2)
    # oshape = (batch_size, sample_size/4*256)
    flatten = Flatten()(relu2)
    # need to store the size of the representation after the convolutions -> needed for deconv later
    hidden_intermediate_enc = Dense(intermediate_dim, name='intermediate_encoding')(flatten)
    hidden_zvalues = Dense(z_size * 2)(hidden_intermediate_enc)
    latent_layer = Lambda(lambda x: x[:, :z_size], output_shape=(z_size,))(hidden_zvalues)

    sampling_object = Sampling(z_size)
    sampling = sampling_object(hidden_zvalues)

    encoder = Model(inputs=input_idx, outputs=[sampling, latent_layer])
    return encoder, sampling_object


def get_decoder(decoder_input, decoder_emb_input, nfilter, sample_size, out_size, nclasses, intermediate_dim, lstm_size):
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
    softmax_auxiliary = Dense(nclasses, activation='softmax', name='auxiliary_softmax_layer')(reshape)

    def remove_last_column(x):
        return x[:, :-1, :]

    padding = ZeroPadding1D(padding=(1, 0))(decoder_emb_input)
    previous_char_slice = Lambda(remove_last_column, output_shape=(sample_size, nclasses))(padding)
    combined_input = concatenate(inputs=[softmax_auxiliary, previous_char_slice], axis=2)
    # MUST BE IMPLEMENTATION 1 or 2
    lstm = LSTM(lstm_size, return_sequences=True, implementation=2)
    recurrent_component = lstm(combined_input)
    final_softmax_layer = Dense(nclasses, activation='softmax', name='final_softmax_layer')
    softmax_final = final_softmax_layer(recurrent_component)

    decoder = Model(inputs=[decoder_input, decoder_emb_input], outputs=softmax_final)

    inference_layer = LSTMStep(lstm, final_softmax_layer, sample_size, nclasses)(softmax_auxiliary)
    decoder_inference = Model(inputs=decoder_input, outputs=inference_layer)

    return decoder, decoder_inference


def get_descriminator(g_in, nfilter, intermediate_dim, step):
    anneal = K.clip(- (0.1/2000.0)*step + 0.1, 0.001, 0.1)

    noise_layer = GaussianNoise(stddev=anneal)(g_in)
    # oshape = (batch_size, sample_size/2, 128)
    conv1 = Conv1D(filters=nfilter, kernel_size=3, strides=2, padding='same')(noise_layer)
    bn1 = BatchNormalization(scale=False)(conv1)
    relu1 = PReLU()(bn1)
    # oshape = (batch_size, sample_size/4, 128)
    conv2 = Conv1D(filters=2 * nfilter, kernel_size=3, strides=2, padding='same')(relu1)
    bn2 = BatchNormalization(scale=False)(conv2)
    relu2 = PReLU()(bn2)
    # oshape = (batch_size, sample_size/4*256)
    flatten = Flatten()(relu2)
    hidden_intermediate_discr = Dense(intermediate_dim, activation='relu')(flatten)

    sigmoid = Dense(1, activation='sigmoid', name='discrimiator_sigmoid')(hidden_intermediate_discr)

    discriminator = Model(inputs=g_in, outputs=[sigmoid, hidden_intermediate_discr])
    #content_embedding = Model(inputs=g_in, outputs=hidden_intermediate_discr)

    return discriminator


def vae_gan_model(config_data, vocab, step):
    z_size = config_data['z_size']
    sample_size = config_data['max_sentence_length']
    nclasses = len(vocab) + 2
    lstm_size = config_data['lstm_size']
    alpha = config_data['alpha']
    intermediate_dim = config_data['intermediate_dim']
    nfilter = 128
    out_size = 200
    eps = 0.001
    anneal_start = 0.0
    anneal_end = anneal_start + 10000.0

    input_idx = Input(batch_shape=(None, sample_size), dtype='int32', name='character_input')

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

    original_sample = one_hot_embeddings(input_idx)

    decoder_input = Input(shape=(z_size,), name='decoder_input')
    decoder_emb_input = Input(shape=(sample_size, nclasses), name='decoder_emb_input')
    dis_input = Input(shape=(sample_size, nclasses))

    encoder, sampling_object = get_encoder(input_idx, original_sample, nfilter, intermediate_dim, z_size)
    decoder, decoder_inference = get_decoder(decoder_input, decoder_emb_input, nfilter, sample_size, out_size, nclasses, intermediate_dim, lstm_size)
    discriminator = get_descriminator(dis_input, nfilter, intermediate_dim, step)

    Z_p, Z = encoder(input_idx)
    X_tilde = decoder(inputs=[Z, original_sample])
    X_p = decoder(inputs=[Z_p, original_sample])

    X_infer = decoder_inference(Z_p)

    #emb_shape = (None, intermediate_dim)
    #orig_embedding = content_embedding(original_sample)
    #rec_embedding = content_embedding(X_tilde)

    dis_orig, orig_embedding = discriminator(original_sample)
    dis_rec_tilde, rec_embedding = discriminator(X_tilde)
    dis_rec_p, p_embedding = discriminator(X_p)



    def vae_cross_ent_loss(args):
        x_truth, x_decoded_final = args
        x_truth_flatten = K.reshape(x_truth, shape=(-1, K.shape(x_truth)[-1]))
        x_decoded_flat = K.reshape(x_decoded_final, shape=(-1, K.shape(x_decoded_final)[-1]))
        cross_ent = K.categorical_crossentropy(target=x_truth_flatten, output=x_decoded_flat)
        cross_ent = K.reshape(cross_ent, shape=(-1, K.shape(x_truth)[1]))
        sum_over_sentences = K.sum(cross_ent, axis=1)
        return sum_over_sentences

    def dis_sim_measure(args):
        oemb, remb = args

        elem_mult = K.square(oemb - remb)

        distance = K.sqrt(K.sum(elem_mult, axis=1) + 1e-5)

        return distance

    def vae_kld_loss(args):
        kl_loss = - 0.5 * K.sum(1 + sampling_object.log_sigma - K.square(sampling_object.mu) - K.exp(sampling_object.log_sigma), axis=-1)
        kld_weight = K.clip((step - anneal_start) / (anneal_end - anneal_start), 0, 1 - eps) + eps
        return kl_loss*kld_weight

    def gan_classification_loss(args):
        discr_x, dirscr_x_tilde, discr_x_p = args

        return - (K.log(K.clip(discr_x, eps, 1-eps)) + K.log(1 - K.clip(dirscr_x_tilde, eps, 1-eps)) + K.log(1 - K.clip(discr_x_p, eps, 1-eps)))

    def generator_loss(args):
        x_fake, = args
        return - K.log(K.clip(x_fake, eps, 1-eps))

    dis_l_loss = Lambda(dis_sim_measure, output_shape=(1,), name='discrimiator_similarity')([orig_embedding, rec_embedding])
    reconstruction_loss = Lambda(vae_cross_ent_loss, output_shape=(1,), name='reconstruction_error')([original_sample, X_p])
    kld_loss = Lambda(vae_kld_loss, output_shape=(1,), name='kld_loss')([original_sample])
    gan_loss = Lambda(gan_classification_loss, output_shape=(1,), name='gan_loss')([dis_orig, dis_rec_tilde, dis_rec_p])
    gen_tilde_loss = Lambda(generator_loss, output_shape=(1,), name='gen_til_loss')([dis_rec_tilde])
    gen_p_loss = Lambda(generator_loss, output_shape=(1,), name='gen_p_loss')([dis_rec_p])

    full_model = Model(inputs=[input_idx], outputs=[dis_sim_measure, kld_loss, gen_p_loss, gen_tilde_loss, gan_loss])
    encoding_train_model = Model(inputs=[input_idx], outputs=[dis_sim_measure, kld_loss])
    decoder_train_model = Model(inputs=[input_idx], outputs=[dis_sim_measure, gen_tilde_loss, gen_p_loss])
    discriminator_train_model = Model(inputs=[input_idx], outputs=[gan_loss])
    discriminator_pretrain_model = Model(inputs=[input_idx], outputs=[dis_orig])

    inference_model = Model(inputs=[input_idx], outputs=[X_infer])

    optimizer = Adadelta(lr=1.0, decay=0.0001, clipnorm=10)

    full_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
    encoding_train_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
    decoder_train_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred, loss_weights=[0.2, 1.0, 1.0])
    discriminator_train_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
    discriminator_pretrain_model.compile(optimizer=optimizer, loss='binary_crossentropy')

    return full_model, encoding_train_model, decoder_train_model, discriminator_train_model, inference_model, encoder, decoder, discriminator, discriminator_pretrain_model


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

    def call(self, x,  **kwargs):
        #(sample_size, batch_size, lstm_size) since we iterate over the samples
        x_shuffled = K.permute_dimensions(x, (1, 0, 2))

        def _step(x_i, states):
            current_word_vec = states[0]
            ins = K.concatenate(tensors=[x_i, current_word_vec], axis=1)

            output, new_states = self.lstm.step(ins, states[1:])

            outsoftmax = self.softmax_layer.call(output)
            # argmax that returns the predicted char
            word_idx = K.argmax(outsoftmax, axis=1)
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

        outputs = K.transpose(K.stack(successive_words))

        return outputs