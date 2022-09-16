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