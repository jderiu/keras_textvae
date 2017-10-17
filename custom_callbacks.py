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


class GANOutputCallback(Callback):
    def __init__(self, test_model, validation_input, frequency, vocabulary, delimiter, fname='logging/test_output'):
        self.validation_input = validation_input
        self.vocabulary = vocabulary
        self.test_model = test_model
        self.frequency = frequency
        self.delimiter = delimiter
        self.fname = fname
        super(GANOutputCallback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0:
            output_text(self.test_model, self.validation_input, self.vocabulary, str(epoch), delimiter=self.delimiter, fname=self.fname)
        self.ep_end_weights = {}

        #loss = logs.get('loss', '-')
        enc_loss = logs.get('enc_loss', '-')
        enc_main_loss = logs.get('enc_main_loss', '-')
        enc_kld_loss = logs.get('enc_kld_loss', '-')
        enc_aux_loss = logs.get('enc_auxiliary_loss', '-')
        enc_z_loss = logs.get('enc_z_loss', '-')

        enc_name_loss = logs.get('enc_name_loss', 0)
        enc_eat_type_loss = logs.get('enc_eat_type_loss', 0)
        enc_price_range_loss = logs.get('enc_price_range_loss', 0)
        enc_feedback_loss = logs.get('enc_feedback_loss_loss', 0)
        enc_near_loss = logs.get('enc_near_loss', 0)
        enc_food_loss = logs.get('enc_food_loss', 0)
        enc_area_loss = logs.get('enc_area_loss', 0)
        enc_family_loss = logs.get('enc_family_loss', 0)

        enc_feature_loss = enc_name_loss + enc_eat_type_loss + enc_price_range_loss + enc_feedback_loss + enc_near_loss + enc_food_loss + enc_area_loss + enc_family_loss
        dis_loss = logs.get('dis_loss', '-')

        val_enc_loss = logs.get('val_enc_loss', '-')
        val_enc_main_loss = logs.get('val_enc_main_loss', '-')
        val_enc_kld_loss = logs.get('val_enc_kld_loss', '-')
        val_enc_aux_loss = logs.get('val_enc_auxiliary_loss', '-')
        val_enc_z_loss = logs.get('val_enc_z_loss', '-')
        val_dis_loss = logs.get('val_dis_loss', '-')
        val_enc_name_loss = logs.get('val_enc_name_loss', 0)
        val_enc_eat_type_loss = logs.get('val_enc_eat_type_loss', 0)
        val_enc_price_range_loss = logs.get('val_enc_price_range_loss', 0)
        val_enc_feedback_loss = logs.get('val_enc_feedback_loss_loss', 0)
        val_enc_near_loss = logs.get('val_enc_near_loss', 0)
        val_enc_food_loss = logs.get('val_enc_food_loss', 0)
        val_enc_area_loss = logs.get('val_enc_area_loss', 0)
        val_enc_family_loss = logs.get('val_enc_family_loss', 0)

        val_enc_feature_loss = val_enc_name_loss + val_enc_eat_type_loss + val_enc_price_range_loss + val_enc_feedback_loss + val_enc_near_loss + val_enc_food_loss + val_enc_area_loss + val_enc_family_loss

        logging.info('TRAINING: Enc Loss: {0: <32}\tEnc Main Loss: {1: <32}\tEnc KLD Loss: {2: <32}\tEnc Aux Loss: {3: <32}\tEnc Z Loss: {4: <32}\tEnc Feature Loss: {5: <32}\tDis Loss: {6: <32}'.format(enc_loss, enc_main_loss, enc_kld_loss, enc_aux_loss, enc_z_loss, enc_feature_loss, dis_loss))
        logging.info('VALIDATION: Enc Loss: {0: <32}\tEnc Main Loss: {1: <32}\tEnc KLD Loss: {2: <32}\tEnc Aux Loss: {3: <32}\tEnc Z Loss: {4: <32}\tEnc Feature Loss: {5: <32}\tDis Loss: {6: <32}'.format(val_enc_loss, val_enc_main_loss, val_enc_kld_loss, val_enc_aux_loss, val_enc_z_loss,val_enc_feature_loss, val_dis_loss))
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
