import keras.backend as K
import numpy as np
from keras.layers import  Lambda, Conv1D, Embedding, Input, BatchNormalization, Activation, Flatten, Dense, GlobalMaxPooling1D, PReLU, Reshape, Conv2DTranspose, concatenate, ZeroPadding1D
from custom_layers.sampling_layer import Sampling
from custom_layers.sem_recurrent import SC_LSTM
from custom_layers.word_dropout import WordDropout
from keras.models import Model
from keras.optimizers import RMSprop, Adadelta, Nadam


def get_encoder(input_idx, input_one_hot_embeddings, nfilter, z_size, intermediate_dim, one_hot_embeddings):
    # oshape = (batch_size, sample_size/2, 128)
    conv1 = Conv1D(
        filters=nfilter,
        kernel_size=3,
        strides=2,
        padding='same'
    )(one_hot_embeddings)
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

    discr_encoder = Model(inputs=one_hot_embeddings, outputs=[sampling, hidden_mean, hidden_log_sigma], name='discr_encoder')

    z_p, z_mean, z_sigma = discr_encoder(input_one_hot_embeddings)
    encoder = Model(inputs=input_idx, outputs=[z_p, z_mean, z_sigma], name='encoder')

    return encoder, discr_encoder


def get_decoder(decoder_input, nclasses, nfilter, sample_out_size, out_size, intermediate_dim, lstm_size, text_idx, text_one_hot, dialogue_act, inputs, step):
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
    logits = Dense(
        nclasses,
        activation='softmax',
        name='auxiliary_softmax_layer'
    )(reshape)
    temperature = 1 / step

    def temperature_log(logits):
        return logits/temperature

    def remove_last_column(x):
        return x[:, :-1, :]

    padding = ZeroPadding1D(padding=(1, 0))(text_one_hot)
    previous_char_slice = Lambda(remove_last_column, output_shape=(sample_out_size, nclasses))(padding)

    temp_layer = Lambda(temperature_log, output_shape=(sample_out_size, nclasses))(logits)
    softmax_auxiliary = Activation('softmax')(temp_layer)

    lstm = SC_LSTM(
        lstm_size,
        nclasses,
        softmax_temperature=temperature,
        generation_only=False,
        condition_on_ptm1=True,
        semantic_condition=True,
        return_da=False,
        return_state=False,
        use_bias=True,
        return_sequences=True,
        implementation=2,
        dropout=0.2,
        recurrent_dropout=0.2,
        sc_dropout=0.2
    )

    recurrent_component = lstm([softmax_auxiliary, previous_char_slice, dialogue_act])
    #output_gen_layer = lstm([softmax_auxiliary, softmax_auxiliary])  # for testing

    decoder_train = Model(inputs=[decoder_input, text_idx] + inputs, outputs=[recurrent_component, softmax_auxiliary], name='decoder_{}'.format('train'))
    decoder_test = Model(inputs=[decoder_input, text_idx] + inputs, outputs=recurrent_component, name='decoder_{}'.format('test'))
    #decoder_train.summary()
    return decoder_train, decoder_test


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


def get_descriminator(g_in, targets, nfilter, intermediate_dim):
    in_conv0 = conv_block(g_in, nfilter)

    hidden_intermediate_discr = Dense(intermediate_dim, activation='relu', name='discr_activation')(in_conv0)

    discr_losses = []
    for target, nlabels, name in targets:
        logits = Dense(nlabels, activation='linear', name='softmax_{}'.format(name))(hidden_intermediate_discr)
        softmax = Activation(activation='softmax')(logits)
        discr_losses.append(softmax)
        #discr_loss = Lambda(cross_ent_loss, output_shape=(1,), name='loss_{}'.format(name))([sigmoid, target])
        #discr_losses.append(discr_loss)

    discriminator = Model(inputs=g_in, outputs=discr_losses, name='discriminator')
    return discriminator


def get_vae_gan_model(config_data, vocab_char, step):
    z_size = config_data['z_size']
    sample_in_size = config_data['max_sentence_len']
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
    name_idx = Input(batch_shape=(None, 2), dtype='float32', name='name_idx')
    eat_type_idx = Input(batch_shape=(None, 4), dtype='float32', name='eat_type_idx')
    price_range_idx = Input(batch_shape=(None, 7), dtype='float32', name='price_range_idx')
    customer_feedback_idx = Input(batch_shape=(None, 7), dtype='float32', name='customer_feedback_idx')
    near_idx = Input(batch_shape=(None, 2), dtype='float32', name='near_idx')
    food_idx = Input(batch_shape=(None, 8), dtype='float32', name='food_idx')
    area_idx = Input(batch_shape=(None, 3), dtype='float32', name='area_idx')
    family_idx = Input(batch_shape=(None, 3), dtype='float32', name='family_idx')
    text_idx = Input(batch_shape=(None, sample_in_size), dtype='int32', name='character_output')
    da_size = 36

    inputs_list = [
        (name_idx, 2, 'name'),
        (eat_type_idx, 4, 'eat_type'),
        (price_range_idx, 7, 'price_range'),
        (customer_feedback_idx, 7, 'feedback'),
        (near_idx, 2, 'near'),
        (food_idx, 8, 'food'),
        (area_idx, 3, 'area'),
        (family_idx, 3, 'family')
    ]
    inputs = [x[0] for x in inputs_list]

    dialogue_act = concatenate(inputs=inputs)

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
    text_one_hot = one_hot_embeddings(text_idx)

    dropped_output_idx = WordDropout(rate=1.0, dummy_word=dropout_word_idx, anneal_step=step, anneal_start=anneal_start, anneal_end=anneal_end)(text_idx)
    dropped_one_hot = one_hot_embeddings(dropped_output_idx)

    # == == == == == =
    # Define Encoder
    # == == == == == =
    enc_dis_input = Input(shape=(sample_in_size, nclasses))
    encoder, discr_encoder = get_encoder(text_idx, text_one_hot, nfilter, z_size, intermediate_dim, enc_dis_input)

    # == == == == == =
    # Define Decoder
    # == == == == == =
    decoder_input = Input(shape=(z_size + da_size,), name='decoder_input')
    decoder_train, decoder_test = get_decoder(decoder_input, nclasses, nfilter, sample_in_size, out_size, intermediate_dim, lstm_size, text_idx, dropped_one_hot, dialogue_act, inputs, step)

    # == == == == == == == =
    # Define Discriminators
    # == == == == == == == =
    dis_input = Input(shape=(sample_in_size, nclasses))
    discriminator = get_descriminator(dis_input, inputs_list, nfilter, intermediate_dim)

    def vae_cross_ent_loss(args):
        x_truth, x_decoded_final = args
        x_truth_flatten = K.reshape(x_truth, shape=(-1, K.shape(x_truth)[-1]))
        x_decoded_flat = K.reshape(x_decoded_final, shape=(-1, K.shape(x_decoded_final)[-1]))
        cross_ent = K.categorical_crossentropy(x_decoded_flat, x_truth_flatten)
        cross_ent = K.reshape(cross_ent, shape=(-1, K.shape(x_truth)[1]))
        sum_over_sentences = K.sum(cross_ent, axis=1)
        return sum_over_sentences

    def vae_kld_loss(args):
        mu, log_sigma = args
        kl_loss = - 0.5 * K.sum(1 + log_sigma - K.square(mu) - K.exp(log_sigma), axis=-1)
        kld_weight = K.clip((step - anneal_start) / (anneal_end - anneal_start), 0, 1 - eps) + eps
        return kl_loss*kld_weight

    def vae_aux_loss(args):
        x_truth, x_decoded_final = args
        x_truth_flatten = K.reshape(x_truth, shape=(-1, K.shape(x_truth)[-1]))
        x_decoded_flat = K.reshape(x_decoded_final, shape=(-1, K.shape(x_decoded_final)[-1]))
        cross_ent = K.categorical_crossentropy(x_decoded_flat, x_truth_flatten)
        cross_ent = K.reshape(cross_ent, shape=(-1, K.shape(x_truth)[1]))
        sum_over_sentences = K.sum(cross_ent, axis=1)
        return alpha*sum_over_sentences

    def enc_discr_dist(args):
        z_mean, z_pred_mean = args
        return K.sum(K.square(z_mean - z_pred_mean), axis=1)

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    def cross_ent_loss(args):
        y_pred, y_true = args
        return K.categorical_crossentropy(y_pred, y_true)

    def da_loss_fun(args):
        da = args[0]
        sq_da_t = K.square(da)
        sum_sq_da_T = K.sum(sq_da_t, axis=1)
        return sum_sq_da_T

    def da_history_loss_fun(args):
        da_t = args[0]
        zeta = 10e-4
        n = 100
        #shape: batch_size, sample_size
        norm_of_differnece = K.sum(K.square(da_t), axis=2)
        n1 = zeta**norm_of_differnece
        n2 = n*n1
        return K.sum(n2, axis=1)

    z_prior, z_mean, z_sigmoid = encoder(text_idx)
    latent_var = concatenate(inputs=[z_prior, dialogue_act], axis=1)
    x_p, x_aux = decoder_train([latent_var, text_idx] + inputs)
    _, z_pred_prior, _ = discr_encoder(x_p)

    dlosses = discriminator(x_p)
    discr_losses = []
    for dloss, (target, nlabel, name) in zip(dlosses, inputs_list):
        discr_loss = Lambda(cross_ent_loss, output_shape=(1,), name='{}'.format(name))([dloss, target])
        discr_losses.append(discr_loss)

    x_argmax = decoder_test([latent_var, text_idx] + inputs)

    #vae_loss
    main_loss = Lambda(vae_cross_ent_loss, output_shape=(1,), name='main')([text_one_hot, x_p])
    kld_loss = Lambda(vae_kld_loss, output_shape=(1,), name='kld')([z_mean, z_sigmoid])
    aux_loss = Lambda(vae_aux_loss, output_shape=(1,), name='auxiliary')([text_one_hot, x_aux])
    encoding_discrimination_loss = Lambda(enc_discr_dist, output_shape=(1,), name='z')([z_mean, z_pred_prior])
    #da_loss = Lambda(da_loss_fun, output_shape=(1,), name='dialogue_act')([da_act_t])
    #da_history_loss = Lambda(da_history_loss_fun, output_shape=(1,), name='dialogue_history')([da_act_history])
    argmax = Lambda(argmax_fun, output_shape=(sample_in_size,))(x_argmax)

    vae_vanilla_train_model = Model(inputs=inputs + [text_idx], outputs=[main_loss, kld_loss, aux_loss])#for pretraining
    vae_vanilla_test_model = Model(inputs=inputs + [text_idx], outputs=argmax)#for pretraining
    vae_model_train = Model(inputs=inputs + [text_idx], outputs=[main_loss, kld_loss, aux_loss, encoding_discrimination_loss] + discr_losses) #for training
    vae_model_test = Model(inputs=inputs + [text_idx], outputs=argmax)

    #discriminator_training
    optimizer_ada = Adadelta(lr=1.0, epsilon=1e-8, rho=0.95, decay=0.0001)

    discr_train_losses = discriminator(text_one_hot)
    discriminator_model = Model(inputs=text_idx, outputs=discr_train_losses)
    discriminator_model.compile(optimizer=optimizer_ada, loss='categorical_crossentropy')

    vae_model_train.compile(optimizer=optimizer_ada, loss=lambda y_true, y_pred: y_pred)
    vae_vanilla_train_model.compile(optimizer=optimizer_ada, loss=lambda y_true, y_pred: y_pred)

    return vae_model_train, vae_model_test, vae_vanilla_train_model, vae_vanilla_test_model, discriminator_model, decoder_test, discriminator