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
# This is a module for creating DenseNets (Huang et al., 
# https://arxiv.org/abs/1608.06993). The module is minimalistic, not much 
# encapsulation takes place. I think this design choice is OK for the purpose 
# of this assignment, since I only use the module to build a single model.
#
# The module can add a dense upsampling block as well a'la Jégou et al. 
# (https://arxiv.org/abs/1611.09326).
#
# Layer and block methods mimic the behavior of Keras layer objects, but are 
# not Keras layer objects (was too lazy to inherit properly).
#
# Simplest way to get the model is to use: 
#     densenet.dense_net(...args, full_model=True).


from . import basic
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Callable, Union


def dense_grid(bottleneck_compression: Union[int, float] = 64,
               growth_rate: int = 32) -> Callable:
    """ Grid block of the DenseNet.
    
    Loosely based on Huang et al., https://arxiv.org/abs/1608.06993.
    
    Args:
        bottleneck_compression: either the number of features in the bottleneck
            layer (if >= 1) or the fraction of features kept in the compression
            layer.
        growth_rate: number of layers added by the block in the concatenation
            layer.   
    
    Returns:
        A function that behaves like a Keras layer.
    """

    def inner(input_):
        input_channels = input_.shape.as_list()[-1]
        if bottleneck_compression < 1:
            bottleneck_features = int(bottleneck_compression * input_channels)
        else:
            bottleneck_features = bottleneck_compression
        compression = basic.conv2d(bottleneck_features, (1, 1), 
                                   activation="linear")
        branch_convolution = basic.conv2d(growth_rate, (3, 3))
        branch_1 = compression(input_)
        branch_1 = branch_convolution(branch_1)
        return layers.Concatenate()([branch_1, input_])

    return inner


def dense_pool(bottleneck_compression: float = 0.5,
               pool_size: int = 2) -> Callable:
    """ Pooling block in DenseNet.
    
    Loosely based on Huang et al., https://arxiv.org/abs/1608.06993.
    
    Args:
        bottleneck_compression: either the number of features in the bottleneck
            layer (if >= 1) or the fraction of features kept in the compression
            layer.
        pool_size: pool size, default is 2 (should not be touched).
        
    Returns:
        A function that behaves like a Keras layer.
    """

    def inner(input_):
        input_channels = input_.shape.as_list()[-1]
        if bottleneck_compression < 1:
            bottleneck_features = int(bottleneck_compression * input_channels)
        else:
            bottleneck_features = bottleneck_compression
        compression = basic.conv2d(bottleneck_features, 
                                   (1, 1), 
                                   activation="linear")
        branch_1 = compression(input_)
        return layers.AveragePooling2D(pool_size=(pool_size, pool_size),
                                       padding="same")(branch_1)

    return inner


def dense_upsampling(output_expansion: float = 2,
                     kernel_size: Tuple[int, int] = (2, 2),
                     stride: int = 2) -> Callable:
    """ Upsampling block in DenseNet.
    
    Loosely based on Huang et al., https://arxiv.org/abs/1608.06993.
    
    Args:
        output_expansion: the number of filters after in the learnt upsampling
            will be this times the number of incoming features.
        kernel_size: kernel size of the transposed convolution.
        stride: stride parameter of the transposed convolution.
        
    Returns:
        A function that behaves like a Keras layer.
    """
    def inner(input_):
        input_channels = input_.shape.as_list()[-1]
        filters = int(input_channels * output_expansion)
        output = layers.Conv2DTranspose(filters,
                                        kernel_size=kernel_size,
                                        strides=(stride, stride),
                                        padding="same")(input_)
        output = layers.Activation("relu")(output)
        return layers.BatchNormalization()(output)

    return inner


def dense_stem(features: int = 64,
               pool: bool = False) -> Callable:
    """ Stem block of DenseNet.
    
    Loosely based on Huang et al., https://arxiv.org/abs/1608.06993.
    
    Args:
        features: the number of filters in the first convolutional layer.
        pool: if True, a 2-wise maxpooling layer will be inserted after
            convolution (based on original paper).
        
    Returns:
        A function that behaves like a Keras layer.
    """
    def inner(input_):
        output = basic.conv2d(features, 
                              kernel_size=(7, 7), 
                              padding="same")(input_)
        if pool:
            output = layers.MaxPool2D(pool_size=(2, 2))(output)
        return output

    return inner


def dense_net(stem_features: int = 64,
              stem_pool: bool = False,
              num_block: int = 3,
              block_growth_rate: int = 32,
              block_bottleneck_features: int = 64,
              num_pool: int = 3,
              pool_bottleneck_compression: float = 0.5,
              upsample: bool = False,
              upsample_output_expansion: float = 1,
              fully_connected_head: bool = False,
              output_classes: int = 3,
              output_steps: int = 3,
              full_model: bool = False,
              num_channels: int = 3) -> Union[keras.models.Model, Callable]:
    """ Full DenseNet model.
    
    Loosely based on Huang et al., https://arxiv.org/abs/1608.06993.
    
    Args:
        ... args are mostly of the building blocks (sorry, should have 
        encapsulated instead), except
        fully_connected_head: convenience argument, if True, inserts
            basic.global_max_head for classifying into output_classes classes.
        full_model: if True, returns a Keras Model with variable input shape
            instead of a function.
        
    Returns:
        A function that behaves like a Keras layer or a Keras Model.
    """
    def inner(input_):
        # stem
        output = dense_stem(features=stem_features,
                            pool=stem_pool)(input_)
        # downsample
        for _ in range(num_pool):
            # grid
            for _ in range(num_block):
                output = dense_grid(
                    bottleneck_compression=block_bottleneck_features,
                    growth_rate=block_growth_rate)(output)
            output = dense_pool(
                bottleneck_compression=pool_bottleneck_compression)(output)
        # upsample
        if upsample:
            for _ in range(num_pool):
                output = dense_upsampling(
                    output_expansion=upsample_output_expansion)(output)
                for _ in range(num_block):
                    output = dense_grid(
                        bottleneck_compression=block_bottleneck_features,
                        growth_rate=block_growth_rate)(output)
        else:
            for _ in range(num_block):
                output = dense_grid(
                    bottleneck_compression=block_bottleneck_features,
                    growth_rate=block_growth_rate)(output)
        # output
        if fully_connected_head:
            output = basic.global_max_head(num_classes=output_classes,
                                           num_steps=output_steps)(output)
        return output

    if full_model:
        input_ = layers.Input(shape=(None, None, num_channels))
        return keras.models.Model(inputs=[input_], outputs=[inner(input_)])
    return inner
