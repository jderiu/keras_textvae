from keras.models import Layer
import keras.backend as K
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class WordDropout(Layer):
    """Applies Word Dropout to the input.

        Word Dropout consists of setting a fraction of the indices to a dummy token. This shoudl prevent the decoder t rely soley on the language model.

        # Arguments
            rate: float between 0 and 1. Fraction of the input units to drop.
            noise_shape: 1D integer tensor representing the shape of the
                binary dropout mask that will be multiplied with the input.
                For instance, if your inputs have shape
                `(batch_size, timesteps, features)` and
                you want the dropout mask to be the same for all timesteps,
                you can use `noise_shape=(batch_size, 1, features)`.
            seed: A Python integer to use as random seed.
        """
    def __init__(self, rate, dummy_word, noise_shape=None, seed=None, **kwargs):
        super(WordDropout, self).__init__(**kwargs)
        self.srng = RandomStreams(seed=np.random.randint(1000000))
        self.p = min(1., max(0., rate))
        self.dummy_word = dummy_word
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, _):
        return self.noise_shape

    def call(self, inputs, training=None):
        if 0. < self.p < 1.:

            def dropped_inputs():
                mask = self.srng.binomial(inputs.shape, p=1 - self.p, dtype='int32')
                return inputs * mask + self.dummy_word * (1 - mask)

            return K.in_train_phase(dropped_inputs, inputs, training=training)

        return inputs

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(WordDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))