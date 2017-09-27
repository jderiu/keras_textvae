import keras.backend as K
from theano import tensor as T
import numpy as np
from keras.layers import ZeroPadding1D, Lambda, Conv1D, Embedding, Input, BatchNormalization, Flatten, Dense, GlobalMaxPooling1D, PReLU, Reshape, Conv2DTranspose, Activation, concatenate, GaussianNoise
from custom_layers.sampling_layer import Sampling
from custom_layers.sem_recurrent import SC_LSTM
from custom_layers.word_dropout import WordDropout
from keras.models import Model
from keras.optimizers import RMSprop, Adadelta, Nadam


def get_encoder(input_idx, input_one_hot_embeddings, nfilter, z_size, intermediate_dim):
    # oshape = (batch_size, sample_size/2, 128)
    conv1 = Conv1D(
        filters=nfilter,
        kernel_size=3,
        strides=2,
        padding='same'
    )(input_one_hot_embeddings)
    bn1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(bn1)
    # oshape = (batch_size, sample_size/4, 128)
    conv2 = Conv1D(
        filters=2 * nfilter,
        kernel_size=3,
        strides=2,
        padding='same'
    )(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(bn2)
    conv3 = Conv1D(
        filters=2 * nfilter,
        kernel_size=3,
        strides=2,
        padding='same',
    )(relu2)
    bn3 = BatchNormalization()(conv3)
    relu3 = Activation('relu')(bn3)
    # oshape = (batch_size, sample_size/4*256)
    flatten = Flatten()(relu3)
    # need to store the size of the representation after the convolutions -> needed for deconv later
    hidden_intermediate_enc = Dense(
        intermediate_dim,
        name='intermediate_encoding'
    )(flatten)
    hidden_mean = Dense(z_size, name='mu')(hidden_intermediate_enc)
    hidden_log_sigma = Dense(z_size, name='sigma')(hidden_intermediate_enc)

    sampling_object = Sampling(z_size)
    sampling = sampling_object([hidden_mean, hidden_log_sigma])

    encoder = Model(inputs=input_idx, outputs=[sampling, hidden_mean, hidden_log_sigma])

    return encoder


def get_decoder(decoder_input, nclasses, nfilter, sample_out_size, out_size, intermediate_dim):
    decoder_input_layer = Dense(intermediate_dim, name='intermediate_decoding')
    hidden_intermediate_dec = decoder_input_layer(decoder_input)
    decoder_upsample = Dense(int(2 * nfilter * sample_out_size / 4))(hidden_intermediate_dec)
    relu_int = PReLU()(decoder_upsample)
    if K.image_data_format() == 'channels_first':
        output_shape = (2 * nfilter, int(sample_out_size / 4), 1)
    else:
        output_shape = (int(sample_out_size / 4), 1, 2 * nfilter)
    reshape = Reshape(output_shape)(relu_int)
    # shape = (batch_size, filters)
    deconv1 = Conv2DTranspose(filters=nfilter, kernel_size=(3, 1), strides=(2, 1), padding='same')(reshape)
    bn3 = BatchNormalization(scale=False)(deconv1)
    relu3 = PReLU()(bn3)
    deconv2 = Conv2DTranspose(filters=out_size,  kernel_size=(3, 1), strides=(2, 1), padding='same')(relu3)
    bn4 = BatchNormalization(scale=False)(deconv2)
    relu4 = PReLU()(bn4)
    reshape = Reshape((sample_out_size, out_size))(relu4)
    softmax_auxiliary = Dense(
        nclasses,
        activation='softmax',
        name='auxiliary_softmax_layer'
    )(reshape)

    decoder_train = Model(inputs=decoder_input, outputs=softmax_auxiliary, name='decoder_{}'.format('train'))
    #decoder_train.summary()
    return decoder_train


def conv_block(input, nfilter):
    conv1 = Conv1D(filters=nfilter, kernel_size=3, strides=2, padding='same')(input)
    bn1 = BatchNormalization(scale=False)(conv1)
    relu1 = PReLU()(bn1)
    # oshape = (batch_size, sample_size/4, 128)
    conv2 = Conv1D(filters=2 * nfilter, kernel_size=3, strides=2, padding='same')(relu1)
    bn2 = BatchNormalization(scale=False)(conv2)
    relu2 = PReLU()(bn2)
    # oshape = (batch_size, sample_size/4*256)
    max_pool = GlobalMaxPooling1D()(relu2)

    return max_pool


def get_descriminator(g_in, g_out, nfilter, intermediate_dim):
    in_conv0 = conv_block(g_in, nfilter)
    in_conv1 = conv_block(g_out, nfilter)

    conv_concat = concatenate(inputs=[in_conv0, in_conv1], axis=1)
    hidden_intermediate_discr = Dense(intermediate_dim, activation='relu', name='discr_activation')(conv_concat)

    sigmoid = Dense(1, activation='tanh', name='discrimiator_sigmoid')(hidden_intermediate_discr)

    discriminator = Model(inputs=[g_in, g_out], outputs=[sigmoid])
    return discriminator


def get_vae_gan_model(config_data, vocab_char, step):
    z_size = config_data['z_size']
    sample_in_size = config_data['max_input_length']
    sample_out_size = config_data['max_output_length']
    nclasses = len(vocab_char) + 2
    # last available index is reserved as start character
    max_idx = max(vocab_char.values())
    dummy_word_idx = max_idx + 1
    dropout_word_idx = max_idx + 2
    word_dropout_rate = config_data['word_dropout_rate']
    lstm_size = config_data['lstm_size']
    alpha = config_data['alpha']
    intermediate_dim = config_data['intermediate_dim']
    nfilter = 128
    out_size = 200
    eps = 0.001

    anneal_start = config_data['anneal_start']
    anneal_end = anneal_start + config_data['anneal_duration']

    # == == == == == =
    # Define Char Input
    # == == == == == =
    input_idx = Input(batch_shape=(None, sample_in_size), dtype='int32', name='character_input')
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
    input_one_hot_embeddings = one_hot_embeddings(input_idx)

    dropped_output_idx = WordDropout(rate=word_dropout_rate, dummy_word=dropout_word_idx)(output_idx)

    one_hot_weights = np.identity(nclasses)
    one_hot_out_embeddings = Embedding(
        input_length=sample_out_size,
        input_dim=nclasses,
        output_dim=nclasses,
        weights=[one_hot_weights],
        trainable=False,
        name='one_hot_out_embeddings'
    )
    output_one_hot_embeddings = one_hot_out_embeddings(dropped_output_idx)

    def remove_last_column(x):
        return x[:, :-1, :]

    padding = ZeroPadding1D(padding=(1, 0))(output_one_hot_embeddings)
    orig_output = Lambda(remove_last_column, output_shape=(sample_out_size, nclasses))(padding)

    # == == == == == =
    # Define Encoder
    # == == == == == =
    encoder = get_encoder(input_idx, input_one_hot_embeddings, nfilter, z_size, intermediate_dim)

    # == == == == == =
    # Define Decoder
    # == == == == == =
    decoder_input = Input(shape=(z_size,), name='decoder_input')
    decoder_train = get_decoder(decoder_input, nclasses, nfilter, sample_out_size, out_size, intermediate_dim)

    # == == == == == == == =
    # Define Discriminators
    # == == == == == == == =
    dis_input = Input(shape=(sample_in_size, nclasses))
    dis_output = Input(shape=(sample_out_size, nclasses))

    discriminator = get_descriminator(dis_input, dis_output, nfilter, intermediate_dim)

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

    def gan_classification_loss(args):
        discr_x, dirscr_xp = args

        return - 0.5*K.log(K.clip(discr_x, eps, 1-eps)) - 0.5*K.log(1 - K.clip(dirscr_xp, eps, 1-eps))

    def generator_loss(args):
        x_fake, = args
        return - K.log(K.clip(x_fake, eps, 1-eps))

    def wasserstein(y_true, y_pred):
        return K.mean(y_true * y_pred)

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    z_prior, z_mean, z_sigmoid = encoder(input_idx)
    x_auxiliary = decoder_train(z_prior)

    #put sc-lst outside of decoder.. some strange problem with disconnected gradients
    lstm = SC_LSTM(
        lstm_size,
        nclasses,
        generation_only=False,
        condition_on_ptm1=True,
        semantic_condition=False,
        return_da=False,
        return_state=False,
        use_bias=True,
        return_sequences=True,
        implementation=2,
        dropout=0.2,
        recurrent_dropout=0.2,
        sc_dropout=0.2
    )

    recurrent_component = lstm([x_auxiliary, orig_output])
    lstm.inference_phase()
    output_gen_layer = lstm([x_auxiliary, x_auxiliary])#for testing

    #vae_loss
    main_loss = Lambda(vae_cross_ent_loss, output_shape=(1,), name='main_loss')([output_idx, recurrent_component])
    kld_loss = Lambda(vae_kld_loss, output_shape=(1,), name='kld_loss')([z_mean, z_sigmoid])
    aux_loss = Lambda(vae_aux_loss, output_shape=(1,), name='auxiliary_loss')([output_idx, x_auxiliary])
    argmax = Lambda(argmax_fun, output_shape=(sample_out_size,))(output_gen_layer)
    vae_model_train = Model(inputs=[input_idx, output_idx], outputs=[main_loss, kld_loss, aux_loss])
    vae_model_test = Model(inputs=input_idx, outputs=argmax)

    #decoder training
    noise_input = Input(batch_shape=(None, z_size), dtype='float32', name='noise_input')
    noise_on_input = GaussianNoise(stddev=1.0)(noise_input)
    noise_model = Model(inputs=[noise_input], outputs=[noise_on_input], name='noise_model')
    noise = noise_model(noise_input)

    x_aux_prior = decoder_train(noise)
    lstm.train_phase = True
    output_gen_layer = lstm([x_aux_prior, x_aux_prior])#recurrent using auxiliary prior
    discr_sigmoid = discriminator([input_one_hot_embeddings, output_gen_layer])
    decoder_discr_model = Model(inputs=[input_idx, noise_input], outputs=discr_sigmoid)

    #decoder test
    x_aux_prior = decoder_train(noise)
    lstm.train_phase = False
    output_gen_layer = lstm([x_aux_prior, x_aux_prior])
    argmax = Lambda(argmax_fun, output_shape=(sample_out_size,))(output_gen_layer)
    decoder_test_model = Model(inputs=noise_input, outputs=argmax)

    #discriminator_training
    discr_input = Input(batch_shape=(None, sample_out_size), dtype='int32', name='discr_output')
    discr_emb = one_hot_out_embeddings(discr_input)

    discr_sigmoid = discriminator([input_one_hot_embeddings, discr_emb])
    discriminator_model = Model(inputs=[input_idx, discr_input], outputs=discr_sigmoid)

    #compile the training models
    optimizer_rms = RMSprop(lr=1e-3, decay=0.0001, clipnorm=10)
    optimizer_ada = Adadelta(lr=1.0, epsilon=1e-8, rho=0.95, decay=0.0001, clipnorm=10)
    optimizer_nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.001)

    vae_model_train.compile(optimizer=optimizer_ada, loss=lambda y_true, y_pred: y_pred)
    decoder_discr_model.compile(optimizer=optimizer_rms, loss=wasserstein)
    discriminator_model.compile(optimizer=optimizer_rms, loss=wasserstein)

    return vae_model_train, vae_model_test, decoder_discr_model, decoder_test_model, discriminator_model, discriminator