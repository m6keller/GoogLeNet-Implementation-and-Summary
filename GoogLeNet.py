import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, concatenate
from keras.models import Model

class GoogLeNet(tf.keras.Model):
    def __init__(self, input, **kwargs):
        super(GoogLeNet, self).__init__(name="GoogLeNet", **kwargs)
        self._input = input

    def inception_module(self, model, f1_conv1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4_pool, name):
        # 1x1 conv
        path1 = Conv2D(filters=f1_conv1, kernel_size=(1, 1), activation='relu', padding='same', name=name+"path_1_conv_2D")(model)

        # 1x1 conv -> 3x3 conv
        path2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1), activation='relu', padding='same', name=name+"path_2_conv_2D_1")(model)
        path2 = Conv2D(filters=f2_conv3, kernel_size=(3, 3), activation='relu', padding='same', name=name+"path_2_conv_2D_2")(path2)

        # 1x1 conv -> 5x5 conv
        path3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1), activation='relu', padding='same', name=name+"path_3_conv_2D_1")(model)
        path3 = Conv2D(filters=f3_conv5, kernel_size=(5, 5), activation='relu', padding='same', name=name+"path_3_conv_2D_2")(path3)

        # 3x3 pool -> 1x1 conv
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=name+"path_4_max_pool")(model)
        path4 = Conv2D(filters=f4_pool, kernel_size=(1, 1), activation='relu', padding='same', name=name+"path_4_conv_2D")(path4)

        output_layer = concatenate([path1, path2, path3, path4], axis = -1)
        return output_layer


    def _conv_1(self):
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name="initial_conv")(self._input)
        return x

    def _max_pool_1(self, x):
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="initial_pool")(x)
        return x

    def _conv_2(self, x):
        x = Conv2D(filters=64, kernel_size=(1, 1), strides= 1, padding='same', activation='relu', name="conv_2")(x)
        x = Conv2D(filters=192, kernel_size=(3,3), padding='same', activation='relu')(x)
        return x

    def _max_pool_2(self, x):
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="max_pool_2")(x)
        return x

    def _inception_3(self, x):
        x = self.inception_module(model=x, f1_conv1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4_pool=32, name="inception_3a")
        x = self.inception_module(model=x, f1_conv1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4_pool=64, name="inception_3b")
        return x


    def _max_pool_3(self, x):
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        return x

    def _inception_4(self, x):
        x = self.inception_module(model=x, f1_conv1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4_pool=64, name="inception_4a")
        x = self.inception_module(model=x, f1_conv1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4_pool=64, name="inception_4b")
        x = self.inception_module(model=x, f1_conv1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4_pool=64, name="inception_4c")
        x = self.inception_module(model=x, f1_conv1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4_pool=64, name="inception_4d")
        x = self.inception_module(model=x, f1_conv1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4_pool=128, name="inception_4e")
        return x

    def _max_pool_4(self, x):
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        return x

    def _inception_5(self, x):
        x = self.inception_module(model=x, f1_conv1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4_pool=128, name="inception_5a")
        x = self.inception_module(model=x, f1_conv1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48, f3_conv5=128, f4_pool=128, name="inception_5b")
        return x

    def _avg_pool(self, x):
        x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
        return x

    def _dropout(self, x):
        x = Dropout(0.4)(x)
        return x

    # def _linear(self, x):
    #     pass

    def _softmax(self, x):
        x = Dense(1000, activation='softmax')(x)
        return x

    def call(self):
        x = self._conv_1(self._input)
        x = self._max_pool_1(x)
        x = self._conv_2(x)
        x = self._max_pool_2(x)
        x = self._inception_3(x)
        x = self._max_pool_3(x)
        x = self._inception_4(x)
        x = self._max_pool_4(x)
        x = self._inception_5(x)
        x = self._avg_pool(x)
        x = self._dropout(x)
        # x = self._linear(x)
        x = self._softmax(x)

        model = Model(input, [x], name='GoogLeNet')
        return model