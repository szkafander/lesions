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
# This module contains a Dataset class and some helper functions to handle the
# HAM 10000 dataset - or any dataset in general, that contains images and some
# auxiliary and target data in form of a metadata file.

from lib import utils
import numpy as np
import os
import pandas as pd
import PIL
import sklearn as sk
from tensorflow import keras
from typing import Union, Sequence, Tuple, Any, Optional


def one_hot_encode(labels: pd.DataFrame,
                   label_order: list) -> np.ndarray:
    """ One-hot encoding for label arrays.
    
    Encodes a label array into a one-hot encoded label array. Why is this 
    needed? I couldn not find an implementation that supports reordering the 
    labels and my Dataset class hacks around .flow_from_dataframe via yielding
    the indices...
    
    Args:
        labels: numpy array holding integer labels. Normally size n, where n is
            the number of samples. Can also other iterables or a pandas Series.
        label_order: a list of string labels in the order to be encoded. For
            example, one_hot_encode with a label_order of ['b', 'c', 'a'] would
            encode the labels np.array(['a', 'a', 'c', 'b']) to
            np.array([[0,0,1], [0,0,1], [0,1,0], [1,0,0]]). This is needed to
            match class label orders with the class_weights keyword argument in
            tf.keras.models.Model.fit_generator.
    
    Returns:
        An array of one-hot encoded labels, normally size n X m, where m is the
            number of classes. Each row has 0's and exactly one 1 value.
    """
    vectors = np.eye(len(label_order)).tolist()
    mapping = {label: vectors[k] for k, label in enumerate(label_order)}
    return np.array([mapping[label] for label in labels])


class Dataset(keras.utils.Sequence):

    def __init__(self,
                 path_images: str,
                 metadata: Union[str, pd.DataFrame],
                 validation_split: float = 0.3,
                 field_idx: str = "index",
                 field_filenames: str = "image_id",
                 field_class: str = "dx",
                 fields_aux: Optional[Tuple[str, ...]] = ("sex", "age"),
                 aux_categorical: Optional[Tuple[bool, ...]] = (True, False),
                 aux_class_order: Optional[Tuple[Sequence, ...]] = (
                         "male", "female"),
                 augmentation: bool = True,
                 rotation_range: int = 90,
                 width_shift_range: float = 0.0,
                 height_shift_range: float = 0.0,
                 random_seed: int = 0,
                 subset: str = "training",
                 batch_size: int = 50,
                 aux_data: bool = True,
                 target_size: Tuple[int, int] = (450, 600),
                 class_order: Optional[Tuple[str, ...]] = None) -> None:
        """ Initializes a Dataset object.
        
        A Dataset object from a path to a folder that contains the
        dataset images and a DataFrame of or path to metadata. The class 
        is not generic, it assumes that usage will be specific to the HAM10000
        dataset. To be honest, this class is not needed, I only wrote it for
        demonstration purposes and to clean up the script a bit via 
        encapsulation. And the example also had a similar class. Note on 
        normalization: the generator method of the class always scales images
        between 0 and 1. I have not included an option for standardization,
        since the dataset seemed quite well normalized in terms of intensity
        already, with no outliers. Class has no support for user IO - if you 
        want to change an attribute a posteriori, re-initialize the object. As
        we re-initialize every time, no performance considerations have been
        made when writing this class. Multi-threaded yielding from the 
        generator is already handled by keras.models.Model.fit_generator and
        keras.preprocessing.image.ImageDataGenerator. Default values are all
        problem-specific for now.
        
        Args:
            path_images: path to a folder that contains the images.
            metadata: path to a metadata csv file or a pandas DataFrame of the
                metadata.
            validation_split: float that specifies the fraction of data held as
                validation data.
            field_idx: name of the field in the metadata DataFrame that holds
                the row indices. Unfortunately cannot use the index column for
                this (thanks Keras).
            field_filenames: name of the field in the metadata DataFrame that
                holds the image filenames.
            field_class: name of the field in the metadata DataFrame that holds
                the supervised classes for the images.
            fields_aux: a sequence of names of fields in the metadata
                DataFrame that hold the additional inputs to serve. Match
                these with the model that consumes the generator!
            aux_catagorical: a list that holds a bool for each element in 
                fields_aux. If the kth element is True in aux_categorical, the 
                kth auxiliary input from fields_aux will be treated as 
                catagorical and one-hot encoded by the generator. Otherwise,
                the input is treated as numerical.
            augmentation: if True, yielded images will be augmented.
            rotation_range: rotation range for augmentation in degrees.
            width_shift_range: range of horizontal translation in augmentation
                in pixels.
            height_shift_range: range of vertical translation in augmentation
                in pixels.
        """
        self.path_images = path_images
        if isinstance(metadata, str):
            metadata = utils.read_metadata(metadata)
        self.metadata = metadata
        self.validation_split = validation_split
        self.field_idx = field_idx
        self.field_filenames = field_filenames
        self.field_class = field_class
        self.fields_aux = fields_aux
        self.aux_class_order = aux_class_order
        self.aux_categorical = aux_categorical
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.aux_data = aux_data
        self.target_size = target_size
        self.class_order = class_order

        # these are kept for the record
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range

        # we store the ImageDataGenerator object in this attribute in case we 
        # need to pull parameters from it later. for an explanation of
        # augmentation params, see the writeup.
        args = {"validation_split": validation_split,
                "rotation_range": rotation_range,
                "width_shift_range": width_shift_range,
                "height_shift_range": height_shift_range,
                "horizontal_flip": True,
                "vertical_flip": True,
                "rescale": 1 / 255}
        self._datagen = keras.preprocessing.image.ImageDataGenerator(**args)

    @property
    def num_data(self):
        """ Returns the number of rows in self.metadata.
        """
        return len(self.metadata)

    @property
    def unique_classes(self) -> np.ndarray:
        """ Returns unique class labels as numpy array.
        """
        if self.class_order is None:
            return self.metadata[self.field_class].unique()
        return self.class_order

    @property
    def class_counts(self) -> pd.Series:
        """ Returns the number of entries for each unique class.
        
        Returns a pandas Series holding the number of rows in self.metadata for
            each unique class in the column self.field_class.
        """
        return self.metadata[self.field_class].value_counts()

    @property
    def class_weights(self) -> np.ndarray:
        """ Balanced class weights for classes.
        
        Returns balanced class weights using sklearn.utils.class_weight. These
        are multipliers for losses to yield balanced learning. The order is the
        same as in self.unique_classes.
        """
        return sk.utils.class_weight.compute_class_weight('balanced',
            self.unique_classes, self.metadata[self.field_class])

    def balance(self,
                mode: str = "upsample",
                aggressiveness: float = 0.5) -> None:
        """ Balances dataset by up- or downsampling.
        
        Upsampling adds replicates of rows in self.metadata for classes that
        do
        not have enough samples so that their final count is agressiveness *
        the count of the most populous class. Note that this is only acceptable
        if we do image augmentation as well on the replicates. Even then, it is
        still hacky. Downsampling removes count * agressiveness entries from 
        the most populous class.
        
        Args:
            aggressiveness: a float that controls the aggressiveness of 
                balancing. The total number of upsampled entries will be 
                aggressiveness * number of entries in most populous class.
        
        Returns:
            None (void, acts on self).
        """
        class_counts = self.class_counts
        pop_class = class_counts.idxmax()
        num_pop_class = int(class_counts[pop_class])
        if mode == "upsampling":
            for klass in self.unique_classes:
                if klass != pop_class:
                    # a little ugly - it just appends randomly sampled 
                    # replicates of rows that hold under-represented classes 
                    # until the number of rows for every class is at least 
                    # num_pop_class * aggressiveness.
                    num_class = int(class_counts[klass])
                    num_rows_to_append = (np.floor(num_pop_class
                                                   * aggressiveness)
                                          .astype(np.int) - num_class)
                    rows_to_append_from = self.metadata[
                            self.metadata[self.field_class] == klass]
                    inds_to_append = np.random.choice(list(range(num_class)),
                                                      size=num_rows_to_append,
                                                      replace=True)
                    self.metadata = self.metadata.append(
                        rows_to_append_from.iloc[inds_to_append,],
                        ignore_index=True)
        elif mode == "downsampling":
            num_rows_to_drop = int(num_pop_class * aggressiveness)
            indices_to_drop = np.random.choice(list(range(num_pop_class)),
                                               size=num_rows_to_drop,
                                               replace=False)
            self.metadata = self.metadata.drop(
                self.metadata.index[indices_to_drop])
        else:
            raise ValueError("Balancing mode must be either 'upsampling' "
                             "or 'downsampling'.")

    def read_and_augment_image(self,
                               filename: str,
                               int_max: int = 255) -> np.ndarray:
        image = np.array(PIL.Image.open(filename).resize(self.target_size))
        image = image / int_max
        if self.augmentation:
            return self._datagen.random_transform(image)
        return image

    def generate(self) -> Tuple[Any, Any]:
        """ Generates data.
        
        This method is supposed to be used multi-threaded with 
        tf.keras.models.Model.fit_generator. Yielding is managed by the base
        class, through the __getitem__ method that calls this. The returned 
        data is a 2-tuple (images, target classes) unless aux_data is 
        requested, in which case ([images, x1, x2,...], target classes) is 
        returned, where x0 is the image data and x1... are the auxiliary data.
        
        Returns:
            A tuple of x, y data.
        """

        indices = np.random.randint(0, high=self.num_data,
                                    size=self.batch_size)

        # read images
        filenames_images = [os.path.join(self.path_images, fn) for fn in
                            self.metadata[self.field_filenames].iloc[indices]]
        x = np.array([self.read_and_augment_image(fn) for fn in
                      filenames_images])

        # one-hot encoded target class labels
        y = one_hot_encode(self.metadata[self.field_class].iloc[indices],
                           self.unique_classes)

        # add auxiliary data if requested
        if self.aux_data:
            aux_inputs = []
            for k, field in enumerate(self.fields_aux):
                aux_input = np.array(self.metadata[field].iloc[indices])
                if self.aux_categorical[k]:
                    aux_input = one_hot_encode(aux_input,
                                               self.aux_class_order[k])
                aux_inputs.append(aux_input)
            x = [x, *aux_inputs]
        return (x, y)

    def __len__(self):
        """ This must be implemented for tf.keras.utils.Sequence. 
        It has no purpose otherwise.
        """
        return 1

    def __getitem__(self, index):
        """ This pulls the values in tf.keras.utils.Sequence. The index is not 
        used, we return data randomly, but is in the method signature.
        """
        return self.generate()
