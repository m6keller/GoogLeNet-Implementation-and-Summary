from GoogLeNet import GoogLeNet
from keras.layers import Input
from keras.models import Model

def test():
    model = GoogLeNet(input = Input(shape=(224, 224, 3)))
    x = model._conv_1()
    assert x.shape == (None, 112, 112, 64)

    x = model._max_pool_1(x)
    assert x.shape == (None, 56, 56, 64)

    x = model._conv_2(x)
    assert x.shape == (None, 56, 56, 192) 

    x = model._max_pool_2(x)
    assert x.shape == (None, 28, 28, 192)

    x = model._inception_3(x)
    assert x.shape == (None, 28, 28, 480)

    x = model._max_pool_3(x)
    assert x.shape == (None, 14, 14, 480)

    x = model._inception_4(x)
    assert x.shape == (None, 14, 14, 832)

    x = model._max_pool_4(x)
    assert x.shape == (None, 7, 7, 832)

    x = model._inception_5(x)
    assert x.shape == (None, 7, 7, 1024)

    x = model._avg_pool(x)
    assert x.shape == (None, 1, 1, 1024)

    x = model._dropout(x)
    assert x.shape == (None, 1, 1, 1024)

    x = model._softmax(x)
    assert x.shape == (None, 1, 1, 1000)
    
    print("All tests passed")