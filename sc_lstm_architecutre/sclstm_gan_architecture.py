import keras.backend as K
import numpy as np
from keras.layers import Lambda, Embedding, Input, concatenate, ZeroPadding1D, Conv1D, BatchNormalization, PReLU, GlobalMaxPooling1D, Activation, Dense

from custom_layers.sem_recurrent import SC_LSTM
from custom_layers.word_dropout import WordDropout
from custom_layers.ctc_decoding_layer import CTC_Decoding_layer
from keras.models import Model
from data_loaders.lex_features_utils import get_lengths

def conv_block(input, nfilter, kernel_size=3):
    conv1 = Conv1D(filters=nfilter, kernel_size=kernel_size, strides=2, padding='same')(input)
    bn1 = BatchNormalization(scale=False)(conv1)
    relu1 = PReLU()(bn1)
    # oshape = (batch_size, sample_size/4, 128)
    conv2 = Conv1D(filters=2 * nfilter, kernel_size=kernel_size, strides=2, padding='same')(relu1)
    bn2 = BatchNormalization(scale=False)(conv2)
    relu2 = PReLU()(bn2)
    # oshape = (batch_size, sample_size/4*256)
    max_pool = GlobalMaxPooling1D()(relu2)

    return max_pool


def conv_block_layered(input, nfilter,nlayers=2, kernel_size=3):
    conv = Conv1D(filters=nfilter, kernel_size=kernel_size, strides=2, padding='same')(input)
    bn = BatchNormalization(scale=False)(conv)
    tmp_relu = PReLU()(bn)

    for layer in range(1, nlayers):
        # oshape = (batch_size, sample_size/2**layer+1, nkernels*2**nlayer)
        conv = Conv1D(filters=2*nfilter, kernel_size=kernel_size, strides=2, padding='same')(tmp_relu)
        bn = BatchNormalization(scale=False)(conv)
        tmp_relu = PReLU()(bn)
        # oshape = (batch_size, sample_size/4*256)
    max_pool = GlobalMaxPooling1D()(tmp_relu)

    return max_pool


def get_descriminator_multitask(g_in, targets, nfilter, intermediate_dim, kernel_size=3):
    in_conv0 = conv_block(g_in, nfilter, kernel_size)

    hidden_intermediate_discr = Dense(intermediate_dim, activation='relu', name='discr_activation')(in_conv0)

    discr_losses = []
    for target, nlabels, name in targets:
        logits = Dense(nlabels, activation='linear', name='hidden_{}'.format(name))(hidden_intermediate_discr)

        softmax = Activation(activation='softmax', name='softmax_{}'.format(name))(logits)
        discr_losses.append(softmax)

    discriminator = Model(inputs=g_in, outputs=discr_losses, name='discriminator')
    return discriminator


def get_discriminator(g_in, nlabel, name, nfilter, hidden_units, kernel_size, nlayers):
    in_conv0 = conv_block_layered(g_in, nfilter, nlayers, kernel_size)

    hidden_intermediate_discr = Dense(hidden_units, activation='relu', name='discr_activation')(in_conv0)

    logits = Dense(nlabel, activation='linear', name='hidden_{}'.format(name))(hidden_intermediate_discr)

    softmax = Activation(activation='softmax', name='softmax_{}'.format(name))(logits)

    discriminator = Model(inputs=g_in, outputs=softmax, name='{}_disc'.format(name))
    return discriminator


def sc_lstm_decoder(text_idx, text_one_hot, dialogue_act, nclasses, sample_out_size, lstm_size, inputs, step):

    def remove_last_column(x):
        return x[:, :-1, :]

    padding = ZeroPadding1D(padding=(1, 0))(text_one_hot)
    previous_char_slice = Lambda(remove_last_column, output_shape=(sample_out_size, nclasses))(padding)

    temperature = 1 / step

    lstm = SC_LSTM(
        lstm_size,
        nclasses,
        softmax_temperature=None,
        generation_only=True,
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

    recurrent_component = lstm([previous_char_slice, dialogue_act])

    decoder = Model(inputs=inputs + [text_idx], outputs=recurrent_component, name='decoder_{}'.format('train'))
    return decoder


def vae_model(config_data, vocab, step):
    sample_out_size = config_data['max_output_length']
    nclasses = len(vocab) + 3
    #last available index is reserved as start character
    lstm_size = config_data['lstm_size']
    max_idx = max(vocab.values())
    dummy_word_idx = max_idx + 1
    dropout_word_idx = max_idx + 1
    nfilter = config_data['nb_filter']
    nlayers = config_data['nlayers']
    filter_length = config_data['filter_length']
    intermediate_dim = config_data['intermediate_dim']
    max_nsentences = config_data['max_nsentences']

    anneal_start = config_data['anneal_start']
    anneal_duration = config_data['anneal_duration']
    alpha = config_data['alpha']

    feature_list = config_data['features']

    # == == == == == =
    # Define Encoder
    # == == == == == =
    name_idx = Input(batch_shape=(None, 2), dtype='float32', name='name_idx')
    eat_type_idx = Input(batch_shape=(None, 4), dtype='float32', name='eat_type_idx')
    price_range_idx = Input(batch_shape=(None, 7), dtype='float32', name='price_range_idx')
    customer_feedback_idx = Input(batch_shape=(None, 7), dtype='float32', name='customer_feedback_idx')
    near_idx = Input(batch_shape=(None, 2), dtype='float32', name='near_idx')
    food_idx = Input(batch_shape=(None, 2), dtype='float32', name='food_idx')
    area_idx = Input(batch_shape=(None, 3), dtype='float32', name='area_idx')
    family_idx = Input(batch_shape=(None, 3), dtype='float32', name='family_idx')

    output_idx = Input(batch_shape=(None, sample_out_size), dtype='int32', name='character_output')

    inputs_list = [
        (name_idx, 2, 'name'),
        (eat_type_idx, 4, 'eat_type'),
        (price_range_idx, 7, 'price_range'),
        (customer_feedback_idx, 7, 'feedback'),
        (near_idx, 2, 'near'),
        (food_idx, 2, 'food'),
        (area_idx, 3, 'area'),
        (family_idx, 3, 'family'),
    ]

    dimensions = get_lengths(config_data)

    if 'nsent' in feature_list:
        x = Input(batch_shape=(None, dimensions['nsent']), dtype='float32', name='nsent_idx')
        inputs_list.append((x, dimensions['nsent'], 'nsent'))

    if 'fout_word_vectors':
        x = Input(batch_shape=(None, dimensions['fout_word_vectors']), dtype='float32', name='fout_word_vectors_idx')
        inputs_list.append((x, dimensions['fout_word_vectors'], 'fout_word_vectors'))

    if 'fout_phrase_vectors':
        x = Input(batch_shape=(None, dimensions['fout_phrase_vectors']), dtype='float32', name='fout_phrase_vectors_idx')
        inputs_list.append((x, dimensions['fout_phrase_vectors'], 'fout_phrase_vectors'))

    if 'fout_pos_vectors':
        x = Input(batch_shape=(None, dimensions['fout_pos_vectors']), dtype='float32', name='fout_pos_vectors_idx')
        inputs_list.append((x, dimensions['fout_pos_vectors'], 'fout_pos_vectors'))

    if 'fword_vectors':
        for i in range(1, max_nsentences):
            fw_idx = Input(batch_shape=(None, dimensions['fword_vectors']), dtype='float32', name='fword_vectors_idx_{}'.format(i))
            inputs_list.append((fw_idx, dimensions['fword_vectors'], 'fword_vectors_{}'.format(i)))

    if 'fphrase_vectors':
        for i in range(1, max_nsentences):
            fw_idx = Input(batch_shape=(None, dimensions['fphrase_vectors']), dtype='float32', name='fphrase_vectors_idx_{}'.format(i))
            inputs_list.append((fw_idx, dimensions['fphrase_vectors'], 'fphrase_vectors_{}'.format(i)))

    if 'fpos_vectors':
        for i in range(1, max_nsentences):
            fw_idx = Input(batch_shape=(None, dimensions['fpos_vectors']), dtype='float32', name='fpos_vectors_idx_{}'.format(i))
            inputs_list.append((fw_idx, dimensions['fpos_vectors'], 'fpos_vectors_{}'.format(i)))

    if 'pos_tag_feature':
        x = Input(batch_shape=(None, dimensions['pos_tag_feature']), dtype='float32', name='pos_tag_feature_idx')
        inputs_list.append((x, dimensions['pos_tag_feature'], 'pos_tag_feature'))

    if 'phrase_tag_feature':
        x = Input(batch_shape=(None, dimensions['phrase_tag_feature']), dtype='float32', name='phrase_tag_feature_idx')
        inputs_list.append((x, dimensions['phrase_tag_feature'], 'phrase_tag_feature'))

    inputs = [x[0] for x in inputs_list]
    word_dropout = WordDropout(rate=1.0, dummy_word=dropout_word_idx, anneal_step=step)(output_idx)

    one_hot_weights = np.identity(nclasses)

    one_hot_out_embeddings = Embedding(
        input_length=sample_out_size,
        input_dim=nclasses,
        output_dim=nclasses,
        weights=[one_hot_weights],
        trainable=False,
        name='one_hot_out_embeddings'
    )

    output_one_hot_embeddings = one_hot_out_embeddings(word_dropout)
    text_one_hot = one_hot_out_embeddings(output_idx)

    dialogue_act = concatenate(inputs=inputs)

    decoder = sc_lstm_decoder(output_idx, output_one_hot_embeddings, dialogue_act, nclasses, sample_out_size, lstm_size, inputs, step)

    dis_input = Input(shape=(sample_out_size, nclasses))

    def vae_cross_ent_loss(args):
        x_truth, x_decoded_final = args
        x_truth_flatten = K.reshape(x_truth, shape=(-1, K.shape(x_truth)[-1]))
        x_decoded_flat = K.reshape(x_decoded_final, shape=(-1, K.shape(x_decoded_final)[-1]))
        cross_ent = K.categorical_crossentropy(x_decoded_flat, x_truth_flatten)
        cross_ent = K.reshape(cross_ent, shape=(-1, K.shape(x_truth)[1]))
        sum_over_sentences = K.sum(cross_ent, axis=1)
        #reconstruction_weight = K.clip((- step + anneal_start + anneal_duration)/anneal_duration, min_value=alpha, max_value=1.0)

        return sum_over_sentences

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    def cross_ent_loss(args):
        y_pred, y_true = args
        return K.categorical_crossentropy(y_pred, y_true)

    x_p = decoder(inputs + [output_idx])

    def remove_first_column(x):
        return x[:, 1:, :]

    discriminators = []
    discr_losses = []
    for target, nlabel, name in inputs_list:
        discriminator = get_discriminator(dis_input, nlabel, name, nfilter, intermediate_dim, kernel_size=filter_length, nlayers=nlayers)
        dloss = discriminator(x_p)
        discr_loss = Lambda(cross_ent_loss, output_shape=(1,), name='{}'.format(name))([dloss, target])
        discr_losses.append(discr_loss)
        discriminators.append((discriminator, name))

    argmax = Lambda(argmax_fun, output_shape=(sample_out_size,))(x_p)

    main_loss = Lambda(vae_cross_ent_loss, output_shape=(1,), name='main')([output_one_hot_embeddings, x_p])
    train_model = Model(inputs=inputs + [output_idx], outputs=[main_loss] + discr_losses[:8])
    test_model = Model(inputs=inputs + [output_idx], outputs=argmax)

    discriminator_models = []
    for discriminator, name in discriminators:
        discr_train_losses = discriminator(text_one_hot)
        discriminator_model = Model(inputs=output_idx, outputs=discr_train_losses, name='{}'.format(name))
        discriminator_models.append(discriminator_model)

    return train_model, test_model, discriminator_models



