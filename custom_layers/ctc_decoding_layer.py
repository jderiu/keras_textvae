from keras.models import Layer
import keras.backend as K
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc



class CTC_Decoding_layer(Layer):
    def __init__(self, sample_out_size, greedy, top_paths, beam_width, dummy_word, **kwargs):
        super(CTC_Decoding_layer, self).__init__(**kwargs)
        self.sample_out_size =sample_out_size
        self.greedy = greedy
        self.top_paths = top_paths
        self.beam_width = beam_width
        self.dummy_word = dummy_word

    def compute_output_shape(self, input_shape):
        single_out_shape = (input_shape[0], self.sample_out_size)
        return self.top_paths * [single_out_shape]

    def compute_mask(self, inputs, mask):
        return self.top_paths * [None]

    def call(self, inputs, training=None):

        y_pred = tf.log(tf.transpose(inputs, perm=[1, 0, 2]) + 1e-8)
        #input_length = tf.to_int32(self.sample_out_size)
        input_length = K.ones_like(inputs[:, 0, 0], dtype='int32')*self.sample_out_size

        if self.greedy:
            (decoded, log_prob) = ctc.ctc_greedy_decoder(
                inputs=y_pred,
                sequence_length=input_length,
                merge_repeated=False
            )
        else:
            (decoded, log_prob) = ctc.ctc_beam_search_decoder(
                inputs=y_pred,
                sequence_length=input_length,
                beam_width=self.beam_width,
                top_paths=self.top_paths,
                merge_repeated=False
            )

        decoded_dense = [tf.sparse_to_dense(st.indices, st.dense_shape, st.values, default_value=-1) for st in decoded]
        dummy_vec = K.ones_like(inputs[:, :, 0], dtype='int64')*self.dummy_word
        conccat_dense = [K.concatenate((d, dummy_vec), axis=1)[:, :self.sample_out_size] for d in decoded_dense]

        return conccat_dense

    def get_config(self):
        config = {
            'sample_out_size': self.sample_out_size,
            'greedy': self.greedy,
            'top_paths': self.top_paths,
            'beam_width': self.beam_width,
        }
        base_config = super(CTC_Decoding_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))