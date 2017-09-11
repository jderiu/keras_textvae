from output_text import output_text, output_lex_text
from keras.callbacks import Callback
import numpy as np
import logging
import keras.backend as K


class StepCallback(Callback):
    def __init__(self, alpha, steps_per_epoch):
        self.alpha = alpha
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        super(StepCallback, self).__init__()

    def on_batch_end(self, batch, logs=None):
        value = self.steps_per_epoch*self.current_epoch + batch
        K.set_value(self.alpha, value)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch


class OutputCallback(Callback):
    def __init__(self, test_model, validation_input, frequency, vocabulary, delimiter, fname='logging/test_output'):
        self.validation_input = validation_input
        self.vocabulary = vocabulary
        self.test_model = test_model
        self.frequency = frequency
        self.delimiter = delimiter
        self.fname = fname
        super(OutputCallback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0:
            output_text(self.test_model, self.validation_input, self.vocabulary, str(epoch), delimiter=self.delimiter, fname=self.fname)
        self.ep_end_weights = {}

        loss = logs.get('loss', '-')
        enc_loss = logs.get('main_loss_loss', '-')
        dec_loss = logs.get('kld_loss_loss', '-')
        dis_loss = logs.get('auxiliary_loss_loss', '-')
        val_loss = logs.get('val_loss', '-')
        val_enc_loss = logs.get('val_main_loss_loss', '-')
        val_dec_loss = logs.get('val_kld_loss_loss', '-')
        val_dis_loss = logs.get('val_auxiliary_loss_loss', '-')

        logging.info('TRAINING: Loss: {}\t Enc Loss: {}\t Dec Loss: {}\tDis Loss: {}'.format(loss, enc_loss, dec_loss, dis_loss))
        logging.info('VALIDATION: Loss: {}\t Enc Loss: {}\t Dec Loss: {}\tDis Loss: {}'.format(val_loss, val_enc_loss, val_dec_loss, val_dis_loss))
        #reset datastructures
        self.ep_begin_weights = {}
        self.ep_end_weights = {}


class LexOutputCallback(Callback):
    def __init__(self, test_model, validation_input, validation_lex, frequency, vocabulary, delimiter, fname='logging/test_output'):
        self.validation_input = validation_input
        self.vocabulary = vocabulary
        self.test_model = test_model
        self.frequency = frequency
        self.delimiter = delimiter
        self.fname = fname
        self.validation_lex = validation_lex

        super(LexOutputCallback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0:
            output_lex_text(self.test_model, self.validation_input, self.validation_lex, self.vocabulary, str(epoch), delimiter=self.delimiter, fname=self.fname)
        self.ep_end_weights = {}

        loss = logs.get('loss', '-')
        enc_loss = logs.get('main_loss_loss', '-')
        dec_loss = logs.get('kld_loss_loss', '-')
        dis_loss = logs.get('auxiliary_loss_loss', '-')
        val_loss = logs.get('val_loss', '-')
        val_enc_loss = logs.get('val_main_loss_loss', '-')
        val_dec_loss = logs.get('val_kld_loss_loss', '-')
        val_dis_loss = logs.get('val_auxiliary_loss_loss', '-')

        logging.info('TRAINING: Loss: {}\t Main Loss: {}\t KLD Loss: {}\tAux Loss: {}'.format(loss, enc_loss, dec_loss, dis_loss))
        logging.info('VALIDATION: Loss: {}\t Main Loss: {}\t KLD Loss: {}\tAux Loss: {}'.format(val_loss, val_enc_loss, val_dec_loss, val_dis_loss))
        #reset datastructures
        self.ep_begin_weights = {}
        self.ep_end_weights = {}

class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered."""

    def __init__(self):
        self.terminated_on_nan = False
        super(TerminateOnNaN, self).__init__()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True
                self.terminated_on_nan = True


def pretrain_discriminator(model, data, vocab):
    nsamples = data.shape[0]
    sample_size = data.shape[1]
    y_orig = np.ones((nsamples, ))
    y_fake = np.zeros((nsamples, ))

    fake_data = np.random.randint(low=0, high=max(vocab.values()), size=(nsamples, sample_size))
    sen_lens = np.random.normal(loc=60, scale=20, size=nsamples)


    train_set = np.vstack((data, fake_data))
    labels = np.vstack((y_orig, y_fake))

    model.fit(train_set, labels, epochs=20)


class MultiModelCheckpoint(Callback):
    def __init__(self, models, period=1):
        super(MultiModelCheckpoint, self).__init__()
        self.models = models
        self.period = period
        self.epochs_since_last_save = 0


    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            for model, filepath in self.models:
                model.save_weights(filepath, overwrite=True)
