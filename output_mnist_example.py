import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

index = 0
target = 9

while index < len(x_train):

    class_type = y_train[index]
    if class_type == target:
        print(class_type)
        with open("mnist_example" + str(target) + ".txt", "w") as f:
            for i in x_train[index]:
                f.write(str(i))
                f.write("\n")

        break
    index += 1

