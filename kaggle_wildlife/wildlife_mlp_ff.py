import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt
import cv2

from PIL import Image
import os,sys


from keras.layers import *
from keras.models import Sequential
from keras.applications import DenseNet121

label_df = pd.read_csv('train.csv')
# test_df = pd.read_csv('test_images.csv')
label_df.head()

def pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        width = ((t,b), (l,r), (0, 0))
    else:
        width = ((t,b), (l,r))
    return pad_width

def pad_and_resize(img_path, dataset, pad=False, desired_size=32):
    img = cv2.imread(f'{dataset}_images/{img_path}.jpg')
    
    if pad:
        width = pad_width(img, max(img.shape))
        padded = np.pad(img, width=width, mode='constant', constant_values=0)
    else:
        padded = img
    
    resized = cv2.resize(padded, (desired_size,)*2).astype('uint8')
    
    return resized

train_resized = []
test_resized = []

num_images = len(label_df['id'])
print_every = 1000

# print("Formatting images...")
# for i, image_id in enumerate(label_df['id']):
#     if i % print_every == 0:
#         print("%2.1f" % (100*i/num_images))
#     train_resized.append(
#         pad_and_resize(image_id, 'train')
#     )

# for image_id in test_df['Id']:
#     test_resized.append(
#         pad_and_resize(image_id, 'test')
#     )

# X_train = np.stack(train_resized)
# # X_test = np.stack(test_resized)

# target_dummies = pd.get_dummies(label_df['category_id'])
# train_label = target_dummies.columns.values
# y_train = target_dummies.values

# print(X_train.shape)
# # print(X_test.shape)
# print(y_train.shape)

# np.save('Resized_Xtrain.npy',X_train)
# # np.save('Resized_Xtest.npy',X_test)
# np.save('Resized_ytrain.npy',y_train)

y_train = np.load('Resized_ytrain.npy')
X_train = np.load('Resized_Xtrain.npy')
# X_test = np.load('../input/reduceddata/wildcam-reduced/X_test.npy')

X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
X_train /= 255
# X_test /= 255

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.33, 0.34, 0.33])
    # return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

X_train = rgb2gray(X_train)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))

print(X_train.shape)
print(y_train.shape)

######### output samples

found_samples = np.zeros(14)

for x,y in zip(X_train, y_train):
    y_index = np.where(y == 1)[0][0]
    if found_samples[y_index] == 1:
        continue
    else:
        found_samples[y_index] = 1
        with open('wildlife_examples/' + str(y_index) + '.txt', 'w') as f:
            for i in x:
                f.write(str(i) + "\n")
        # np.save('wildlife_examples/' + str(y_index) + '.npy',y_index)

quit()

#########

NUM_CLASSES = 14


# dense_network = DenseNet121(input_shape = (32, 32, 3), include_top = False, classes = 1000)
# model = Sequential()
# model.add(dense_network)
# model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.5))
# model.add(Dense(NUM_CLASSES, activation='softmax'))


#### PUT THIS CODE ON A REPO                  
#### Put experiment case studies in the paper

model = Sequential()
# model.add(Dense(100, activation='tanh', input_shape=(32,32,)))
model.add(Dense(250, activation='sigmoid', input_shape=(1024,)))
model.add(Dropout(0.1))
model.add(Dense(200, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(150, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):

    # def Metrics(val_data):
    #     self.validation_data = val_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_pred = self.model.predict(X_val)

        y_pred_cat = keras.utils.to_categorical(
            y_pred.argmax(axis=1),
            num_classes=NUM_CLASSES
        )
        _val_f1 = f1_score(y_val, y_pred_cat, average='macro')
        _val_recall = recall_score(y_val, y_pred_cat, average='macro')
        _val_precision = precision_score(y_val, y_pred_cat, average='macro')

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        print((f"val_f1: {_val_f1:.4f}"
               f" — val_precision: {_val_precision:.4f}"
               f" — val_recall: {_val_recall:.4f}"))

        return

f1_metrics = Metrics()

import keras
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor='val_acc', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)

# print(X_train.shape)
# print(y_train.shape)

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=128,
    epochs=40,
    validation_split=0.2
)
    # callbacks=[f1_metrics],

model.save("wildlife_model.h5", save_format='h5')

fig = plt.subplots(figsize=(8,8))
plt.plot(history.history['loss'],color='g')
plt.plot(history.history['val_loss'],color='r')
plt.legend(['training','validation'])
plt.show()

# fig = plt.subplots(figsize=(8,8))
# plt.plot(history.history['acc'],color='g')
# plt.plot(history.history['val_acc'],color='r')
# plt.legend(['training','validation'])
# plt.show()
