# -*- coding: utf-8 -*-
# Copyright (c) 2019 Paul Lucas
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This module contains a single method that trains a model. While not much is 
# in here, I refactored this into a module to get rid of boilerplate in the 
# main script and in the future this module might grow anyways.

from lib import dataset
import pandas as pd
from tensorflow import keras
from typing import Tuple


history = keras.callbacks.History()
early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", mode="max", patience=5, verbose=1)
lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3,
                                                  verbose=1)
callbacks = [history, early_stopping, lr_reducer]

def compile_and_train(
        model: keras.models.Model, 
        train_gen: dataset.Dataset, 
        validate_gen: dataset.Dataset, 
        model_path: str, 
        learning_rate: float = 0.001,
        num_shows: int = 5000,
        multithreading: bool = True,
        num_workers: int = 20
    ) -> Tuple[keras.models.Model, keras.callbacks.History]:
    """ Trains a model and return the trained model and the training history.
    
    Compiles the model with the requested learning rate with the Adam 
    optimizer. Then trains the model with the starting learning rate. Includes
    early stopping and learning rate scheduling. The model is continuously
    saved to model_path. The best model is reloaded and returned after 
    training.
    
    Args:
        model: a keras Model to train.
        train_gen: a dataset.Dataset object that holds the training data.
        validate_gen: a dataset.Dataset object that holds the validation data.
        model_path: the file path to the checkpointed model.
        learning_rate: the initial learning rate for the optimizer.
        num_shows: proportional to the number of unique images shown to the 
            model at each epoch. With sequential sampling, this would be
            exactly the number of images shown. With random sampling, it is
            at most the number of unique images shown.
    """
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=["categorical_accuracy"])
    checkpoint = keras.callbacks.ModelCheckpoint(model_path, verbose=1, 
                                                 save_best_only=True)
    history = model.fit_generator(
            train_gen, 
            steps_per_epoch=num_shows/train_gen.batch_size, 
            epochs=100, 
            verbose=1, 
            use_multiprocessing=multithreading,
            workers=num_workers,
            class_weight=train_gen.class_weights,
            validation_data=validate_gen,
            validation_steps=num_shows/validate_gen.batch_size,
            callbacks=[*callbacks, checkpoint]
        )
    model = keras.models.load_model(model_path)
    pd.DataFrame(history.history).to_csv(model_path+"_history.csv")
    return model, history