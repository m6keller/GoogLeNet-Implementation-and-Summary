from main import inception_module as Inception_block

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, concatenate

 # input layer 
input_layer = Input(shape = (224, 224, 3))

# First Convolutional Layer
# x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input)

# # First Max Pooling Layer
# x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
# assert x.shape == (None, 56, 56, 64)

# x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
# assert x.shape == (None, 56, 56, 192)

# x = Conv2D(filters = 192, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)
# assert x.shape == (None, 28, 28, 192)

# x = MaxPooling2D(pool_size= (3,3), strides = 2)(x)
# assert x.shape == (None, 28, 28, 256)


# convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
X = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', activation='relu')(input_layer)
print(X.shape)
# max-pooling layer: pool_size = (3,3), strides = 2
X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)
print(X.shape)
# convolutional layer: filters = 64, strides = 1
X = Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)
print(X.shape)
# convolutional layer: filters = 192, kernel_size = (3,3)
X = Conv2D(filters = 192, kernel_size = (3,3), padding = 'same', activation = 'relu')(X)
print(X.shape)
# max-pooling layer: pool_size = (3,3), strides = 2
X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)
print(X.shape)
# # 1st Inception block
# X = Inception_block(X, f1_conv1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)
# print(X.shape)
# # 2nd Inception block
# X = Inception_block(X, f1_conv1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)
# print(X.shape)