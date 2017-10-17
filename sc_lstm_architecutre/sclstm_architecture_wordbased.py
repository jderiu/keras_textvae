import keras.backend as K
import numpy as np
import os
from keras.layers import Lambda, Embedding, Input, concatenate, ZeroPadding1D, TimeDistributed, Dense

from custom_layers.semantically_conditioned_lstm import WCLSTM
from custom_layers.word_dropout import WordDropout
from keras.models import Model


def vae_model(config_data, vocab, step):
    sample_out_size = config_data['max_output_length']
    nclasses = len(vocab) + 3
    #last available index is reserved as start character
    lstm_size = config_data['lstm_size']
    max_idx = len(vocab)
    dummy_word_idx = max_idx + 1
    dropout_word_idx = max_idx + 2
    top_paths = 10

    embeddings = np.load(os.path.join(config_data['vocab_path'], 'embeddings.npy'))
    embedding_size = embeddings.shape[1]
    nclasses = embeddings.shape[0]

    freq_weights = np.ones(shape=(dropout_word_idx, 1))
    total_words = 0
    for idx, freq in vocab.values():
        freq_weights[int(idx)] = freq
        total_words += freq

    freq_embedding = K.variable(freq_weights, dtype='float32', name='frequency_idx')

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
    word_dropout = WordDropout(rate=.5, dummy_word=dropout_word_idx, anneal_step=step, anneal_start=5000, anneal_end=10000)(output_idx)

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

    distributed_embeddings = Embedding(
        input_length=sample_out_size,
        input_dim=nclasses,
        output_dim=embedding_size,
        weights=[embeddings],
        trainable=True
    )

    droppouted_distributed_embeddings = distributed_embeddings(word_dropout)
    embeddings_correct = distributed_embeddings(output_idx)

    dialogue_act = concatenate(inputs=inputs)

    def remove_last_column(x):
        return x[:, :-1, :]

    padding = ZeroPadding1D(padding=(1, 0))(droppouted_distributed_embeddings)
    previous_char_slice = Lambda(remove_last_column, output_shape=(sample_out_size, embedding_size))(padding)

    #combined_input = concatenate(inputs=[softmax_auxiliary, previous_char_slice], axis=2)
    #MUST BE IMPLEMENTATION 1 or 2
    lstm = WCLSTM(
        lstm_size,
        embedding_size,
        return_da=True,
        return_state=False,
        use_bias=True,
        return_sequences=True,
        implementation=2,
        dropout=0.5,
        recurrent_dropout=0.5,
        sc_dropout=0.5
    )
    recurrent_component, last_da, da_array = lstm([previous_char_slice, dialogue_act])

    timedistributed = TimeDistributed(Dense(units=nclasses, activation='softmax'))
    recurrent_decoding = timedistributed(recurrent_component)

    def vae_cross_ent_loss(args):
        x_truth, x_decoded_final = args
        x_truth_flatten = K.reshape(x_truth, shape=(-1, K.shape(x_truth)[-1]))
        x_decoded_flat = K.reshape(x_decoded_final, shape=(-1, K.shape(x_decoded_final)[-1]))

        cross_ent = K.categorical_crossentropy(x_decoded_flat, x_truth_flatten)
        cross_ent = K.reshape(cross_ent, shape=(-1, K.shape(x_truth)[1]))
        sum_over_sentences = K.sum(cross_ent, axis=1)
        return sum_over_sentences

    def variation_loss(args):
        x_decoded_final, = args
        x_decoded_argmax = K.argmax(x_decoded_final, axis=2)  # shape bs, sample size
        x_frequencies = K.squeeze(K.gather(freq_embedding, K.cast(x_decoded_argmax, 'int32')), axis=2)

        x_prob = x_frequencies/total_words

        return K.mean(x_prob, axis=1)

    def word_emb_reconstruction_error(args):
        emb_truth, emb_pred = args
        mean_squared_differnece = K.mean(K.square(emb_truth - emb_pred), axis=2)
        return K.sum(mean_squared_differnece)

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

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    argmax = Lambda(argmax_fun, output_shape=(sample_out_size,))(recurrent_decoding)
    #beams = CTC_Decoding_layer(sample_out_size, False, top_paths, 50, dummy_word_idx)(recurrent_decoding)

    main_loss = Lambda(vae_cross_ent_loss, output_shape=(1,), name='main')([output_one_hot_embeddings, recurrent_decoding])
    variation_loss = Lambda(variation_loss, output_shape=(1,), name='variation')([recurrent_decoding])
    da_loss = Lambda(da_loss_fun, output_shape=(1,), name='da')([last_da])
    da_history_loss = Lambda(da_history_loss_fun, output_shape=(1,), name='da_history')([da_array])
    word_reconstruction = Lambda(word_emb_reconstruction_error, output_shape=(1,), name='word_rec')([embeddings_correct, recurrent_component])

    train_model = Model(inputs=inputs + [output_idx], outputs=[main_loss, da_loss, da_history_loss, word_reconstruction, variation_loss])
    test_model = Model(inputs=inputs + [output_idx], outputs=[argmax])

    return train_model, test_model



