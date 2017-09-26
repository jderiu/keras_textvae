from keras.layers import Layer
import keras.backend as K


class Sampling(Layer):

    def __init__(self, z_size, **kwargs):
        self.z_size = z_size
        super(Sampling, self).__init__(**kwargs)

    def call(self, z, **kwargs):
        mu = z[0]
        log_sigma = z[1]
        eps = K.random_normal(shape=K.shape(mu))

        z_out = mu + K.exp(0.5*log_sigma)*eps
        #store to make them accessible to the loss later on
        self.mu = mu
        self.log_sigma = log_sigma
        return z_out

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2

        assert input_shape[0][1] == self.z_size
        return input_shape[0][0], self.z_size

