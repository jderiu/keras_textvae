from keras.layers import Layer
import keras.backend as K


class Sampling(Layer):

    def __init__(self, z_size, **kwargs):
        self.z_size = z_size
        super(Sampling, self).__init__(**kwargs)

    def call(self, z):
        mu = z[:, :self.z_size]
        log_sigma = z[:, self.z_size:]
        eps = K.random_normal(shape=mu.shape)

        z_out = mu + K.exp(0.5*log_sigma)*eps
        #store to make them accessible to the loss later on
        self.mu = mu
        self.log_sigma = log_sigma
        return z_out

    def compute_output_shape(self, input_shape):
        assert input_shape[1] == 2*self.z_size
        return input_shape[0], self.z_size

