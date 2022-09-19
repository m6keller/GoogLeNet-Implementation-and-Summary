from main import inception_module

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, concatenate

import pytest

def test_one():
    assert 1 == 1

def test_shape():
    # GoogLeNet Model
    input = Input(shape=(224, 224, 3))

    # First Convolutional Layer
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input)
    assert (x.shape == (None, 112, 112, 64))
    # First Max Pooling Layer
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    assert x.shape == (None, 56, 56, 64)

    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    assert x.shape == (None, 56, 56, 192)

    x = Conv2D(filters = 192, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)
    assert x.shape == (None, 28, 28, 192)

    x = MaxPooling2D(pool_size= (3,3), strides = 2)(x)
    assert x.shape == (None, 28, 28, 256)


    x = inception_module(model=x, f1_conv1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4_pool=32)
    assert x.shape == (None, 56, 56, 256)
    
    x = inception_module(model=x, f1_conv1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4_pool=64)
    assert x.shape == (None, 56, 56, 480)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    assert x.shape == (None, 28, 28, 480)

    x = inception_module(model=x, f1_conv1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4_pool=64)
    assert x.shape == (None, 28, 28, 512)
    x = inception_module(model=x, f1_conv1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4_pool=64)
    assert x.shape == (None, 28, 28, 512)
    x = inception_module(model=x, f1_conv1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4_pool=64)
    assert x.shape == (None, 28, 28, 512)
    x = inception_module(model=x, f1_conv1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4_pool=64)
    assert x.shape == (None, 28, 28, 528)
    x = inception_module(model=x, f1_conv1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4_pool=128)
    assert x.shape == (None, 28, 28, 832)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    assert x.shape == (None, 14, 14, 832)

    x = inception_module(model=x, f1_conv1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4_pool=128)
    assert x.shape == (None, 14, 14, 832)
    x = inception_module(model=x, f1_conv1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48, f3_conv5=128, f4_pool=128)
    assert x.shape == (None, 14, 14, 1024)

    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
    assert x.shape == (None, 8, 8, 1024)

    x = Dropout(0.4)(x)
    assert x.shape == (None, 8, 8, 1024)
    x = Dense(1000, activation='softmax')(x)
    assert x.shape == (None, 8, 8, 1000)
  
    model = Model(input, [x], name='GoogLeNet')
    model.summary()



