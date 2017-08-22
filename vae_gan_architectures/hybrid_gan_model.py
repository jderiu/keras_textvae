import keras.backend as K
import numpy as np
from keras.layers import Lambda, Conv1D, Embedding, Input, BatchNormalization, Flatten, \
    Dense, GlobalMaxPooling1D, GaussianNoise, PReLU

from keras.models import Model
from keras.optimizers import RMSprop, Adadelta, Nadam
from os.path import join
from vae_gan_architectures.siamese_vae_gan import vae_model


def get_descriminator(g_in, nfilter, intermediate_dim, step):
    #anneal = K.clip(- (1/20000.0)*step + 1.0, 0.001, 1.0)

    #noise_layer = GaussianNoise(stddev=anneal)(g_in)
    # oshape = (batch_size, sample_size/2, 128)
    conv1 = Conv1D(filters=nfilter, kernel_size=3, strides=2, padding='same')(g_in)
    bn1 = BatchNormalization(scale=False)(conv1)
    relu1 = PReLU()(bn1)
    # oshape = (batch_size, sample_size/4, 128)
    conv2 = Conv1D(filters=2 * nfilter, kernel_size=3, strides=2, padding='same')(relu1)
    bn2 = BatchNormalization(scale=False)(conv2)
    relu2 = PReLU()(bn2)
    # oshape = (batch_size, sample_size/4*256)
    max_pool = GlobalMaxPooling1D()(relu2)
    hidden_intermediate_discr = Dense(intermediate_dim, activation='relu', name='discr_activation')(max_pool)

    sigmoid = Dense(1, activation='sigmoid', name='discrimiator_sigmoid')(hidden_intermediate_discr)

    discriminator = Model(inputs=g_in, outputs=[sigmoid])
    discriminator.summary()
    return discriminator


def get_vae_gan_model(config_data, vocab_char, step):
    sample_size = config_data['max_sentence_length']
    nchars = len(vocab_char) + 2
    # last available index is reserved as start character
    lstm_size = config_data['lstm_size']
    alpha = config_data['alpha']
    intermediate_dim = config_data['intermediate_dim']

    embedding_path = join(config_data['vocab_word_path'], 'embedding_matrix.npy')
    embedding_matrix = np.load(open(embedding_path, 'rb'))
    nclasses = embedding_matrix.shape[0]
    emb_dim = embedding_matrix.shape[1]
    nfilter = 128
    out_size = 200
    eps = 0.001

    # == == == == == =
    # Define Char Input
    # == == == == == =
    input_idx_char = Input(batch_shape=(None, sample_size), dtype='int32', name='character_input')

    one_hot_weights = np.identity(nchars)
    #oshape = (batch_size, sample_size, nclasses)
    one_hot_embeddings = Embedding(
        input_length=sample_size,
        input_dim=nchars,
        output_dim=nchars,
        weights=[one_hot_weights],
        trainable=False,
        name='one_hot_embeddings'
    )
    original_sample_char = one_hot_embeddings(input_idx_char)

    # == == == == == =
    # Define Word Input
    # == == == == == =
    input_idx_word = Input(batch_shape=(None, sample_size), dtype='int32', name='word_input')

    # one_hot_weights = np.identity(nclasses)
    # oshape = (batch_size, sample_size, nclasses)
    word_embeddings = Embedding(
        input_length=sample_size,
        input_dim=nclasses,
        output_dim=emb_dim,
        weights=[embedding_matrix],
        trainable=False,
        name='word_embeddings'
    )
    original_sample_word = word_embeddings(input_idx_word)

    vae_train_model, vae_test_model, aux_char_model, aux_word_model, final_output_model = vae_model(config_data,input_idx_char, input_idx_word, original_sample_char, original_sample_word, vocab_char, step)

    # == == == == == == == =
    # Define Discriminators
    # == == == == == == == =
    dis_char_input = Input(shape=(sample_size, nchars))
    dis_word_input = Input(shape=(sample_size, emb_dim))

    discr_main = get_descriminator(dis_char_input, nfilter, intermediate_dim, step)
    discr_aux_char = get_descriminator(dis_char_input, nfilter, intermediate_dim, step)
    discr_aux_word = get_descriminator(dis_word_input, nfilter, intermediate_dim, step)

    X_main = final_output_model([input_idx_char, input_idx_word])
    X_aux_char = aux_char_model([input_idx_char, input_idx_word])
    X_aux_word = aux_word_model([input_idx_char, input_idx_word])

    dis_orig_char_main = discr_main(original_sample_char)
    dis_main = discr_main(X_main)

    dis_orig_char_aux = discr_aux_char(original_sample_char)
    dis_aux_char = discr_aux_char(X_aux_char)

    dis_orig_word_aux = discr_aux_word(original_sample_word)
    dis_aux_word = discr_aux_word(X_aux_word)

    def gan_classification_loss(args):
        discr_x, dirscr_xp = args

        return - 0.5*K.log(K.clip(discr_x, eps, 1-eps)) - 0.5*K.log(1 - K.clip(dirscr_xp, eps, 1-eps))

    def generator_loss(args):
        x_fake, = args
        return - K.log(K.clip(x_fake, eps, 1-eps))

    #Generator Losses
    gen_main_loss = Lambda(generator_loss, output_shape=(1,), name='gen_main')([dis_main])
    gen_aux_char_loss = Lambda(generator_loss, output_shape=(1,), name='gen_aux_char')([dis_aux_char])
    gen_aux_word_loss = Lambda(generator_loss, output_shape=(1,), name='gen_aux_word')([dis_aux_word])

    #GAN losses
    gan_main_loss = Lambda(gan_classification_loss, output_shape=(1,), name='GAN_main')([dis_orig_char_main, dis_main])
    gan_aux_char_loss = Lambda(gan_classification_loss, output_shape=(1,), name='GAN_aux_char')([dis_orig_char_aux, dis_aux_char])
    gan_aux_word_loss = Lambda(gan_classification_loss, output_shape=(1,), name='GAN_aux_word')([dis_orig_word_aux, dis_aux_word])

    vae_gan_model = Model(inputs=[input_idx_char, input_idx_word], outputs=[gen_main_loss,gen_aux_char_loss ,gen_aux_word_loss])
    discriminator_model = Model(inputs=[input_idx_char, input_idx_word], outputs=[gan_main_loss, gan_aux_char_loss, gan_aux_word_loss])

    #compile the training models
    optimizer_rms = RMSprop(lr=0.0003, decay=0.0001, clipnorm=10)
    optimizer_ada = Adadelta(lr=1.0, epsilon=1e-8, rho=0.95, decay=0.0001, clipnorm=10)
    optimizer_nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.001)

    vae_train_model.compile(optimizer=optimizer_nadam, loss=lambda y_true, y_pred: y_pred)
    vae_gan_model.compile(optimizer=optimizer_nadam, loss=lambda y_true, y_pred: y_pred)
    discriminator_model.compile(optimizer=optimizer_nadam, loss=lambda y_true, y_pred: y_pred)

    discriminators = [discr_main, discr_aux_char, discr_aux_word]

    return vae_train_model, vae_gan_model, discriminator_model, discriminators, vae_test_model