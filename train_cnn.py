from architecture import get_cnn
import os
import logging
import keras.backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from final_evaluation import compute_f1_score

if K._BACKEND == 'theano':
    from evaluation_metrics import evaluation_metrics_theano as evaluation_metrics
else:
    from evaluation_metrics import evaluation_metrics_tf as evaluation_metrics


def train_cnn(config_data, training_set, valid_set1, valid_set2, sample_weight=None, shuffle=False):
    model = get_cnn(config_data)
    model.load_weights(config_data['base_model_path'])

    # == == == == == =
    # Set up Training
    # == == == == == =
    options = config_data['optimizer']
    optimizer = optimizers.Adadelta(lr=options['lr'], rho=options['rho'], epsilon=options['epsilon'])
    metrics = [evaluation_metrics.f1_score_semeval]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    early_stopping = EarlyStopping(monitor='val_f1_score_semeval', patience=20, verbose=1, mode='max')
    model_fname = config_data['tmp_cnn_model_path']
    model_checkpoint = ModelCheckpoint(filepath=model_fname, verbose=1, save_best_only=True,
                                       monitor='val_f1_score_semeval', mode='max')

    batch_size = config_data['batch_size']
    nb_epochs = config_data['nb_epochs']

    # == == =
    # Train
    # == == =
    model.fit(
        x=training_set[0],
        y=training_set[1],
        batch_size=batch_size,
        sample_weight=sample_weight,
        validation_data=(valid_set1[0], valid_set1[1]),
        nb_epoch=nb_epochs,
        verbose=1,
        shuffle=shuffle,
        callbacks=[early_stopping, model_checkpoint]
    )

    # == == == =
    # Evaluate
    # == == == =

    weights_path = config_data['tmp_cnn_model_path']
    logging.info('Load Trained Model')
    model.load_weights(weights_path)
    scores = model.evaluate(valid_set2[0], valid_set2[1], batch_size=1000)

    f1_semeval = compute_f1_score(model, valid_set2[0], valid_set2[1])
    logging.info('Val2 Loss: {}, F1 Score: {}'.format(scores[0], f1_semeval))
    fitness = f1_semeval
    return fitness