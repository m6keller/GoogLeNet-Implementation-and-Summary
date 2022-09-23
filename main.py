from GoogLeNet import GoogLeNet
from keras.layers import Input

def main():
    model = GoogLeNet(input = Input(shape=(224, 224, 3)))
    model.call()

if __name__ == '__main__':
    main()

