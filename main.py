from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, concatenate

def inception_module(model, f1_conv1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4_pool):

    # 1x1 conv
    path1 = Conv2D(filters=f1_conv1, kernel_size=(1, 1), activation='relu', padding='same')(model)

    # 1x1 conv -> 3x3 conv
    path2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1), activation='relu', padding='same')(model)
    path2 = Conv2D(filters=f2_conv3, kernel_size=(3, 3), activation='relu', padding='same')(path2)

    # 1x1 conv -> 5x5 conv
    path3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1), activation='relu', padding='same')(model)
    path3 = Conv2D(filters=f3_conv5, kernel_size=(5, 5), activation='relu', padding='same')(path3)

    # 3x3 pool -> 1x1 conv
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(model)
    path4 = Conv2D(filters=f4_pool, kernel_size=(1, 1), activation='relu', padding='same')(path4)

    output_layer = concatenate([path1, path2, path3, path4], axis = -1)

    return output_layer


# GoogLeNet
def googlenet(input):
    # First Convolutional Layer
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input)

    # First Max Pooling Layer
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    x = inception_module(model=x, f1_conv1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4_pool=32)
    x = inception_module(model=x, f1_conv1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4_pool=64)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(model=x, f1_conv1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4_pool=64)
    x = inception_module(model=x, f1_conv1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4_pool=64)
    x = inception_module(model=x, f1_conv1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4_pool=64)
    x = inception_module(model=x, f1_conv1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4_pool=64)
    x = inception_module(model=x, f1_conv1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4_pool=128)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_module(model=x, f1_conv1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4_pool=128)
    
    x = inception_module(model=x, f1_conv1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48, f3_conv5=128, f4_pool=128)
    
    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)

    x = Dropout(0.4)(x)

    x = Dense(1000, activation='softmax')(x)
    
    # GoogLeNet Model
    model = Model(input, [x], name='GoogLeNet')
    return model

def main():
    return googlenet(input = Input(shape=(224, 224, 3)))

if __name__ == '__main__':
    main()