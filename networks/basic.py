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
# This module contains basic DNN building blocks and a vanilla DCNN
# implementation. All methods return inner functions that mimic the behavior of
# Keras layers in a shallow way. A Keras model is returned by methods that 
# accept a full_model keyword argument and if that argument is set to True.

import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple, Callable, Union, Optional


def global_max_head(num_classes: int = 3,
                    num_steps: int = 3,
                    num_units: Optional[int] = None,
                    batchnorm: bool = True,
                    aux_inputs: Optional[
                        Tuple[tf.Tensor, ...]] = None) -> Callable:
    """ Global maximum pooling 'head' that can be placed on top of existing
    models.
    
    Global maxpooling reduces tensors feature-wise, returning the highest 
    activation. Global pooling discards spatial localization information, thus
    it is used with fully convolutional architectures. It introduces
    translation invariance for the same reason. The output of the head is a
    logistic layer for classification. An MLP connects the global maxpooled
    features with the logistic layer.
    
    Args:
        num_classes: number of classes in the final logistic layer.
        num_steps: the number of hidden layers in the MLP.
        num_units: optional, the number of units in the first MLP layer. If not
            provided, num_units in the first layer is calculated by geometric
            scaling based on the incoming features.
        batchnorm: if True, batch normalization layers are inserted after the
            MLP layers' activation.
        aux_inputs: an optional tuple of Tensorflow Tensor objects that are
            merged into the first MLP layer.
    """
    def inner(input_):
        output = layers.GlobalMaxPooling2D()(input_)
        if aux_inputs is not None:
            output = layers.Concatenate()([output, *aux_inputs])
        compression = ((output.shape.as_list()[-1] / num_classes)
                       ** (-1 / num_steps))
        for i in range(num_steps - 1):
            if i == 0:
                if num_units is None:
                    num_features = int(output.shape.as_list()[-1]
                                       * compression)
                else:
                    num_features = num_units
            else:
                num_features = int(output.shape.as_list()[-1] * compression)
            output = layers.Dense(num_features, activation="relu")(output)
            if batchnorm:
                output = layers.BatchNormalization(momentum=0.9)(output)
        output = layers.Dense(num_classes, activation="softmax")(output)
        return output

    return inner


def conv2d(filters: int = 64,
           kernel_size: Tuple[int, int] = (3, 3),
           padding: str = "same",
           strides: Tuple[int, int] = (1, 1),
           batchnorm: bool = True,
           activation: str = "relu",
           upsample: bool = False) -> Callable:
    """ Basic convolutional layer with optional batchnorm and upsampling.
    
    Returns a function that behaves like a Keras layer.
    
    Args:
        filters: number of convolution filters.
        kernel_size: tuple of kernel size.
        padding: padding keyword argument of tf.keras.layers.Conv2D.
        strides: strides keyword argument of tf.keras.layers.Conv2D.
        batchnorm: if True, adds a batch normalization layer after the
            activation layer.
        upsample: if True, adds transposed convolution instead of regular
            convolution ("learnable upsampling").
    """

    def inner(input_):
        if upsample:
            output = layers.Conv2DTranspose(filters,
                                            kernel_size,
                                            strides=strides,
                                            padding=padding)(input_)
        else:
            output = layers.Conv2D(filters,
                                   kernel_size,
                                   strides=strides,
                                   padding=padding)(input_)
        if activation is not None:
            output = layers.Activation(activation)(output)
        if batchnorm:
            output = layers.BatchNormalization()(output)
        return output

    return inner


def conv_net(num_filters: int = 16,
             kernel_shape: Tuple = (3, 3),
             num_block: int = 1,
             num_pool: int = 3,
             pool_expansion: float = 2,
             upsample: bool = False,
             fully_connected_head: bool = False,
             output_classes: int = 3,
             output_steps: int = 3,
             full_model: bool = False,
             batchnorm: bool = True,
             num_channels: int = 3) -> Union[Callable, tf.keras.models.Model]:
    """ Returns a DCNN.
    
    The returned DCNN is a Tensorflow Tensor unless full_model is True. In that
    case, a Keras Model is returned with a variable input shape.
    
    Args:
        num_filters: number of convolutional features in the first 
            convolutional layer. The number of features in consequent layers 
            are calculated so that the number of tensor elements is preserved
            if pool_expansion = 2.
        kernel_shape: convolution kernel shape.
        num_block: number of convolutional layers between two pooling layers.
        num_pool: number of pooling layers.
        pool_expansion: the number of filters in convolutional layers is 
            multiplied by this amount after each pooling layer.
        upsample: if True, a bunch of upsampling blocks are added on top of the
            pooling blocks, symmetrically to the downsampling.
        fully_connected_head: if True, adds a simple global maxpooling head
            with no auxiliary inputs.
        output_classes: number of output classes in the final logistic layer
            if fully_connected_head is True.
        output_steps: num_steps argument of global_max_head if 
            fully_connected_head is True.
        full_model: if True, method returns a Keras Model with variable input
            shape instead of a Tensorflow Tensor.
        batchnorm: if True, batch normalization layers are inserted after
            convolutional layers' activation.
        num_channels: number of channels in the input. Only relevant if
            full_model is True.
    """
    def inner(input_):
        for j in range(num_pool):
            for i in range(num_block):
                if i == 0 and j == 0:
                    output = layers.Conv2D(num_filters * pool_expansion ** (j),
                                           kernel_shape,
                                           padding="same")(input_)
                else:
                    output = layers.Conv2D(num_filters * pool_expansion ** (j),
                                           kernel_shape,
                                           padding="same")(output)
                output = layers.Activation("relu")(output)
                if batchnorm:
                    output = layers.BatchNormalization(momentum=0.9)(output)
            output = layers.AveragePooling2D(pool_size=(2, 2), 
                                             padding="same")(output)
        for _ in range(num_block):
            output = layers.Conv2D(num_filters * pool_expansion ** (num_pool),
                                   kernel_shape,
                                   padding="same")(output)
            output = layers.Activation("relu")(output)
            if batchnorm:
                output = layers.BatchNormalization(momentum=0.9)(output)
        if upsample:
            for j in range(num_pool):
                output = layers.UpSampling2D()(output)
                for _ in range(num_block):
                    # num_units is calculated by a stupid geometric scaling
                    # rule. I read this somewhere in a neural network design
                    # book. This is for simplicity, IMHO I do not attribute too
                    # much importance to this.
                    output = layers.Conv2D(
                            num_filters*pool_expansion**(num_pool-j-1),
                            kernel_shape,
                            padding="same")(output)
                    output = layers.Activation("relu")(output)
                    if batchnorm:
                        output = layers.BatchNormalization(
                                momentum=0.9)(output)
        if fully_connected_head:
            output = global_max_head(num_classes=output_classes,
                                     num_steps=output_steps,
                                     batchnorm=batchnorm)(output)
        return output

    if full_model:
        inp_ = layers.Input(shape=(None, None, num_channels))
        return tf.keras.models.Model(inputs=[inp_], outputs=[inner(inp_)])
    return inner
