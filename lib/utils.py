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
# This module contains helper functions not particularly related to any 
# specific group of functions.

from lib import dataset
import itertools
from tensorflow import keras
import math
import matplotlib.pyplot as pl
import numpy as np
import os
import pandas as pd
import requests
import sklearn as sk
from sklearn import model_selection
import tensorflow as tf
import tqdm
from typing import Tuple, Union
import zipfile


def download_data(url: str,
                  file_path: str,
                  block_size: int = 1024) -> None:
    """ Downloads data from url and writes it to file_path.
    
    Downloads data from a url and writes it to file_path locally. Shows 
    progress bar in the console. This is my version of:
    https://sumit-ghosh.com/articles/python-download-progress-bar/.
    
    Args:
        url: the url to download from
        file_path: a string that specifies the path to the file to be written
    
    Returns:
        None (void)
    """
    print(f"Downloading data from {url}...")
    with open(file_path, 'wb') as file:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        for data in tqdm.tqdm(
                response.iter_content(block_size),
                total=math.ceil(total_size // block_size),
                unit='KB',
                unit_scale=True):
            file.write(data)


def unzip_data(file_path: str,
               target_path: str) -> None:
    """ Unzips a .zip file.
    
    Unzips a .zip file. Should be used with the downloaded dataset files. I 
    have not tested this on MacOS. No progress bar, sorry.
    
    Args:
        file_path: a string that specifies the path to the file to be unzipped
        target_path: a string specifying the output directory where the 
            unzipped contents will be placed
    
    Returns:
        None (void)
    """
    print(f"Unzipping {file_path} to {target_path}...")
    with zipfile.ZipFile(file_path, "r") as zip_file:
        zip_file.extractall(target_path)
    
    
def make_folder(folder: str) -> None:
    """ Creates a folder if does not exist.
    
    Args:
        folder: the folder to create.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def collect_predictions(
        model: Union[str, keras.models.Model],
        test_gen: dataset.Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Loads a model and collects predictions.
    
    Args:
        model: a Keras model or a path to a Keras model.
        test_gen: generator for test data.
    
    Returns:
        A tuple of predictions and true values.
    """
    if isinstance(model, keras.models.Model):
        model = keras.models.load_model(model)
    pred, true = np.empty((0, 7)), np.empty((0,7))
    for i in tqdm.tqdm(range(1000)):
        data = test_gen.generate()
        pred = np.append(pred, model.predict(data[0]), axis=0)
        true = np.append(true, data[1], axis=0)
    return pred, true


def top_k_accuracies(preds, true):
    """ Returns the 3 first top-k accuracies.
    
    Args:
        preds: numpy array of predictions.
        true: numpy array of true values.
    
    Returns:
        A list of the first 3 top-k accuracies.
    """
    acc = [keras.metrics.top_k_categorical_accuracy(true, preds, k=k+1)
           .eval(session=tf.Session())
           for k in range(3)]
    return acc
    
    
def confusion_matrix(preds: np.ndarray,
                     true: np.ndarray,
                     normalize: bool = False) -> np.ndarray:
    """ Computes confusion matrix given predictions and true values.
    
    Args:
        preds: numpy array of predictions.
        true: numpy array of true values.
        normalize: if True, the returned confusion matrix is normalized.
    
    Returns:
        A numpy array, the confusion matrix.
    """
    cm = sk.metrics.confusion_matrix(true.argmax(axis=1),
                                     preds.argmax(axis=1))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


def setup_problem(
        folder_project: str, 
        download: bool = True
    ) -> Tuple[str, str, str, pd.DataFrame]:
    """ Convenience function that sets up the problem.
    
    Downloads and unzips all data and returns a loaded metadata DataFrame. If
    download = False, does not download and unzip data and assumes that data
    are already in folder_project/data/ and folder_project/data/images.
    
    Args:
        project_folder: the full path to the project folder.
    
    Returns:
        Pandas DataFrame holding the metadata.
    """
    
    make_folder(folder_project)
    folder_data = os.path.join(folder_project, "data")
    make_folder(folder_data)
    folder_images = os.path.join(folder_data, "images")
    make_folder(folder_images)
    folder_models = os.path.join(folder_data, "models")
    make_folder(folder_models)
    
    if download:
        url_base = "https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/"
        files = ("HAM10000_metadata.csv", 
                 "HAM10000_images_part_1.zip", 
                 "HAM10000_images_part_2.zip")
        
        for file in files:
            download_data(url_base+file, os.path.join(folder_data, file))
    
        for file in files[1:]:
            unzip_data(os.path.join(folder_data, file), folder_images)
        
    return (folder_data, 
            folder_images, 
            folder_models,
            read_metadata(os.path.join(folder_data, "HAM10000_metadata.csv")))


def setup_data_generators(
        metadata, 
        folder_images,
        field_class = "dx", 
        test_size = 0.1,
        validation_size = 0.2,
        aux_data = False,
        augment = True,
        batch_size = 50,
        balancing = True,
        seed = None
    ) -> Tuple[dataset.Dataset, dataset.Dataset, dataset.Dataset]:
    if seed is None:
        seed = np.random.randint(0, 1e7)
    train_data, test_data, _, _ = model_selection.train_test_split(
            metadata, metadata[field_class], test_size=0.1,
            stratify=metadata[field_class], random_state=seed)
    train_data, validate_data, _, _ = model_selection.train_test_split(
            train_data, train_data[field_class], test_size=0.2,
            stratify=train_data[field_class], random_state=seed)
    
    # set up image generators
    get_dataset = (lambda data, aux_data, class_order=None, augment=True: 
        dataset.Dataset(folder_images, data, target_size=(300,225), 
                        augmentation=augment, aux_data=aux_data, 
                        batch_size=batch_size, class_order=class_order))
    
    # sync class order with training generator
    train_gen = get_dataset(train_data, False)
    test_gen = get_dataset(test_data, False, 
                           class_order=train_gen.unique_classes,
                           augment=False)
    validate_gen = get_dataset(validate_data, False, 
                               class_order=train_gen.unique_classes)
    
    # balance datasets
    if balancing:
        train_gen.balance(mode="upsampling", aggressiveness=0.7)
        test_gen.balance(mode="upsampling", aggressiveness=0.7)
        validate_gen.balance(mode="upsampling", aggressiveness=0.7)
    
    return train_gen, test_gen, validate_gen


def read_metadata(path_metadata: str) -> pd.DataFrame:
    """ Reads metadata from a csv file.
    
    Reads metadata from a csv file and returns a pandas DataFrame. Delimiter is
    hard-coded for the moment. Can only treat .csv files for now. Hardcoded for
    the HAM 10000 metadata file. Does some data cleaning.
    
    Args:
        path_metadata: the path to the metadata file.
    
    Returns:
        A pandas DataFrame with the read metadata.
    """
    metadata = pd.read_csv(path_metadata, delimiter=",")
    metadata = metadata.dropna()
    metadata = metadata[metadata.sex != "unknown"]
    metadata.image_id = metadata.image_id.astype(str) + ".jpg"
    metadata["index"] = range(len(metadata))
    return metadata


def plot_confusion_matrix(predictions: np.ndarray,
                          true_values: np.ndarray,
                          classes: Tuple,
                          normalize: bool = False,
                          cmap=pl.cm.Blues) -> None:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Code is from https://github.com/uyxela/Skin-Lesion-Classifier.
    
    Args:
        predictions: np.ndarray of predictions (one-hot encoded). Size mXn,
            where m is number of samples, n is number of classes.
        true_values: same for the true values.
        classes: a tuple of class name strings in the correct order.
        normalize: if True, confusion matrix will be normalized.
        cmap: matplotlib colormap to use.
    """
    cm = confusion_matrix(predictions, true_values, normalize)

    pl.imshow(cm, interpolation="nearest", cmap=cmap)
    pl.colorbar()
    tick_marks = np.arange(len(classes))
    pl.xticks(tick_marks, classes, rotation=45)
    pl.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pl.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    pl.ylabel("true label")
    pl.xlabel("predicted label")
    pl.tight_layout()
