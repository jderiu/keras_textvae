import keras.backend as K
import numpy as np
from keras.layers import Lambda, Conv1D, Conv2DTranspose, Embedding, Input, BatchNormalization, Activation, Flatten, \
    Dense, Reshape, concatenate, ZeroPadding1D, GlobalAveragePooling1D, PReLU, LSTM

from custom_layers.sem_recurrent import SC_LSTM
from keras.models import Model
from theano import tensor as T



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
    name_idx = Input(batch_shape=(None, 2), dtype='float32', name='name_idx')
    eat_type_idx = Input(batch_shape=(None, 4), dtype='float32', name='eat_type_idx')
    price_range_idx = Input(batch_shape=(None, 7), dtype='float32', name='price_range_idx')
    customer_feedback_idx = Input(batch_shape=(None, 7), dtype='float32', name='customer_feedback_idx')
    near_idx = Input(batch_shape=(None, 2), dtype='float32', name='near_idx')
    food_idx = Input(batch_shape=(None, 8), dtype='float32', name='food_idx')
    area_idx = Input(batch_shape=(None, 3), dtype='float32', name='area_idx')
    family_idx = Input(batch_shape=(None, 3), dtype='float32', name='family_idx')
    output_idx = Input(batch_shape=(None, sample_out_size), dtype='int32', name='character_output')

    inputs = [name_idx, eat_type_idx, price_range_idx, customer_feedback_idx, near_idx, food_idx, area_idx, family_idx]

    one_hot_weights = np.identity(nclasses)

    one_hot_out_embeddings = Embedding(
        input_length=sample_out_size,
        input_dim=nclasses,
        output_dim=nclasses,
        weights=[one_hot_weights],
        trainable=False,
        name='one_hot_out_embeddings'
    )

    output_one_hot_embeddings = one_hot_out_embeddings(output_idx)

    dialogue_act = concatenate(inputs=inputs)

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    def remove_last_column(x):
        return x[:, :-1, :]

    padding = ZeroPadding1D(padding=(1, 0))(output_one_hot_embeddings)
    previous_char_slice = Lambda(remove_last_column, output_shape=(sample_out_size, nclasses))(padding)

    #combined_input = concatenate(inputs=[softmax_auxiliary, previous_char_slice], axis=2)
    #MUST BE IMPLEMENTATION 1 or 2
    lstm = SC_LSTM(
        lstm_size,
        nclasses,
        generation_only=True,
        condition_on_ptm1=True,
        return_da=False,
        return_state=False,
        use_bias=True,
        semantic_condition=True,
        return_sequences=True,
        implementation=2,
        dropout=0.2,
        recurrent_dropout=0.2,
        sc_dropout=0.2
    )
    recurrent_component = lstm([previous_char_slice, dialogue_act])

    lstm.inference_phase()

    output_gen_layer = lstm([previous_char_slice, dialogue_act])

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

    def da_loss_fun(args):
        da = args[0]
        sq_da_t = K.square(da)
        sum_sq_da_T = K.sum(sq_da_t, axis=1)
        return sum_sq_da_T

    def identity_loss(y_true, y_pred):
        return y_pred

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    argmax = Lambda(argmax_fun, output_shape=(sample_out_size,))(output_gen_layer)

    main_loss = Lambda(vae_cross_ent_loss, output_shape=(1,), name='main_loss')([output_idx, recurrent_component])
    #da_loss = Lambda(da_loss_fun, output_shape=(1,), name='dialogue_act_loss')([state])

    #output_gen_layer = LSTMStep(lstm, final_softmax_layer, sample_out_size, nclasses)(softmax_auxiliary)

    train_model = Model(inputs=inputs + [output_idx], outputs=[main_loss])
    test_model = Model(inputs=inputs + [output_idx], outputs=[argmax])

    return train_model, test_model



