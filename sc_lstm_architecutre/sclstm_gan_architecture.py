import keras.backend as K
import numpy as np
from keras.layers import Lambda, Embedding, Input, concatenate, ZeroPadding1D, Conv1D, BatchNormalization, PReLU, GlobalMaxPooling1D, Activation, Dense

from custom_layers.sem_recurrent import SC_LSTM
from custom_layers.word_dropout import WordDropout
from custom_layers.ctc_decoding_layer import CTC_Decoding_layer
from keras.models import Model


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
        logits = Dense(nlabels, activation='linear', name='hidden_{}'.format(name))(hidden_intermediate_discr)

        softmax = Activation(activation='softmax', name='softmax_{}'.format(name))(logits)
        discr_losses.append(softmax)

    discriminator = Model(inputs=g_in, outputs=discr_losses, name='discriminator')
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
        return_da=True,
        return_state=False,
        use_bias=True,
        return_sequences=True,
        implementation=2,
        dropout=0.2,
        recurrent_dropout=0.2,
        sc_dropout=0.2
    )

    recurrent_component, last_da, da_array = lstm([previous_char_slice, dialogue_act])

    decoder = Model(inputs=inputs + [text_idx], outputs=[recurrent_component, last_da, da_array], name='decoder_{}'.format('train'))
    return decoder


def vae_model(config_data, vocab, step):
    sample_out_size = config_data['max_output_length']
    nclasses = len(vocab) + 3
    #last available index is reserved as start character
    lstm_size = config_data['lstm_size']
    max_idx = max(vocab.values())
    dummy_word_idx = max_idx + 1
    dropout_word_idx = max_idx + 1
    top_paths = 10

    l2_regularizer = None
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
    fw_idx = Input(batch_shape=(None, 40), dtype='float32', name='fw_idx')
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
        (fw_idx, 40, 'fw'),
    ]

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
    discriminator = get_descriminator(dis_input, inputs_list, 128, 200)

    def vae_cross_ent_loss(args):
        x_truth, x_decoded_final = args
        x_truth_flatten = K.reshape(x_truth, shape=(-1, K.shape(x_truth)[-1]))
        x_decoded_flat = K.reshape(x_decoded_final, shape=(-1, K.shape(x_decoded_final)[-1]))
        cross_ent = K.categorical_crossentropy(x_decoded_flat, x_truth_flatten)
        cross_ent = K.reshape(cross_ent, shape=(-1, K.shape(x_truth)[1]))
        sum_over_sentences = K.sum(cross_ent, axis=1)
        return sum_over_sentences

    def da_loss_fun(args):
        da = args[0]
        sq_da_t = K.square(da)
        sum_sq_da_T = K.sum(sq_da_t, axis=1)
        return sum_sq_da_T

    def da_history_loss_fun(args):
        da_hp1, da_h = args
        zeta = 10e-4
        n = 100
        #shape: batch_size, sample_size
        sum_sq_difference = K.sum(K.square(da_hp1 - da_h), axis=2)

        n1 = zeta**sum_sq_difference[:, :-1]
        n2 = n*n1
        return K.sum(n2, axis=1)


    def identity_loss(y_true, y_pred):
        return y_pred

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    def cross_ent_loss(args):
        y_pred, y_true = args
        return K.categorical_crossentropy(y_pred, y_true)

    x_p, da_T, da_h = decoder(inputs + [output_idx])

    def remove_first_column(x):
        return x[:, :-1, :]

    padding = ZeroPadding1D(padding=(0, 1))(da_h) #shape:: bs, sample size, da_dim
    da_hp1 = Lambda(remove_first_column, output_shape=(sample_out_size, nclasses))(padding)

    dlosses = discriminator(x_p)
    discr_losses = []
    for dloss, (target, nlabel, name) in zip(dlosses, inputs_list):
        discr_loss = Lambda(cross_ent_loss, output_shape=(1,), name='{}'.format(name))([dloss, target])
        discr_losses.append(discr_loss)

    argmax = Lambda(argmax_fun, output_shape=(sample_out_size,))(x_p)

    main_loss = Lambda(vae_cross_ent_loss, output_shape=(1,), name='main')([output_one_hot_embeddings, x_p])
    da_loss = Lambda(da_loss_fun, output_shape=(1,), name='dialogue_act')([da_T])
    da_history_loss = Lambda(da_history_loss_fun, output_shape=(1,), name='dialogue_history')([da_hp1, da_h])

    train_model = Model(inputs=inputs + [output_idx], outputs=[main_loss, da_loss, da_history_loss] + discr_losses)
    test_model = Model(inputs=inputs + [output_idx], outputs=argmax)

    discr_train_losses = discriminator(text_one_hot)
    discriminator_model = Model(inputs=output_idx, outputs=discr_train_losses)
    #discriminator_model.compile(optimizer=optimizer_ada, loss='categorical_crossentropy')

    return train_model, test_model, discriminator_model



