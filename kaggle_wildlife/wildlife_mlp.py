'''
Modified from from:
https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
'''


# # 1) Convert activations to sigmoid
# # 2) Update repository
# # 3) Notify frank
# # 4) Writeup the case study and experiement flow

# # from __future__ import print_function

# # import keras
# # from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import RMSprop
# from keras.optimizers import Adam
# from keras import Model

# from tensorflow import nn

# import os

# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt

# import cv2
# import torch
# from tqdm import tqdm_notebook
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models
# from sklearn.model_selection import train_test_split
# from torchvision import transforms


# batch_size = 64
# IMG_SIZE = 64
# N_EPOCHS = 10
# ID_COLNAME = 'file_name'
# ANSWER_COLNAME = 'category_id'
# TRAIN_IMGS_DIR = 'train_images/'
# TEST_IMGS_DIR = 'test_images/'

# # the data, split between train and test sets
# # (x_train, y_train), (x_test, y_test) = mnist.load_data()

# train_df_all = pd.read_csv('train.csv')
# train_df_all.head()


# train_df, test_df = train_test_split(train_df_all[[ID_COLNAME, ANSWER_COLNAME]],
#                                      test_size = 0.15,                                     
#                                      shuffle = True
#                                     )
# CLASSES_TO_USE = train_df_all['category_id'].unique()
# NUM_CLASSES = len(CLASSES_TO_USE)

# CLASSMAP = dict(
#     [(i, j) for i, j
#      in zip(CLASSES_TO_USE, range(NUM_CLASSES))
#     ]
# )
# REVERSE_CLASSMAP = dict([(v, k) for k, v in CLASSMAP.items()])

# class FF_model(Model):

#   def __init__(self):
#     super(FF_model, self).__init__()
#     self.dense1 = Dense(4, activation=nn.relu)
#     self.dense2 = Dense(5, activation=nn.softmax)

#   def call(self, inputs):
#     x = self.dense1(inputs)
#     return self.dense2(x)

# model = FF_model()

# # model = Sequential()
# # model.add(Dense(40, activation='tanh', input_shape=(IMG_SIZE,IMG_SIZE)))
# # model.add(Dropout(0.2))
# # model.add(Dense(40, activation='tanh'))
# # model.add(Dropout(0.2))
# # model.add(Dense(NUM_CLASSES, activation='softmax'))

# # model = Model(
# #     inputs=model.inputs,
# #     outputs=[layer.output for layer in model.layers],
# # )



# normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

# train_augmentation = transforms.Compose([
#     transforms.Resize((IMG_SIZE,IMG_SIZE)),
#     transforms.ToTensor(),
#     normalizer,
# ])

# val_augmentation = transforms.Compose([
#     transforms.Resize((IMG_SIZE,IMG_SIZE)),
#     transforms.ToTensor(),
#     normalizer,
# ])

# class IMetDataset(Dataset):
    
#     def __init__(self,
#                  df,
#                  images_dir,
#                  n_classes = NUM_CLASSES,
#                  id_colname = ID_COLNAME,
#                  answer_colname = ANSWER_COLNAME,
#                  label_dict = CLASSMAP,
#                  transforms = None
#                 ):
#         self.df = df
#         self.images_dir = images_dir
#         self.n_classes = n_classes
#         self.id_colname = id_colname
#         self.answer_colname = answer_colname
#         self.label_dict = label_dict
#         self.transforms = transforms
    
#     def __len__(self):
#         return self.df.shape[0]
    
#     def __getitem__(self, idx):
#         cur_idx_row = self.df.iloc[idx]
#         img_id = cur_idx_row[self.id_colname]
#         img_name = img_id # + self.img_ext
#         img_path = os.path.join(self.images_dir, img_name)
        
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(img)
        
#         if self.transforms is not None:
#             img = self.transforms(img)
        
#         if self.answer_colname is not None:              
#             label = torch.zeros((self.n_classes,), dtype=torch.float32)
#             label[self.label_dict[cur_idx_row[self.answer_colname]]] = 1.0

#             return img, label
        
#         else:
#             return img, img_id

# train_dataset = IMetDataset(train_df, TRAIN_IMGS_DIR, transforms = train_augmentation)
# test_dataset = IMetDataset(test_df, TRAIN_IMGS_DIR, transforms = val_augmentation)

# BS = 24

# train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=2, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)

# def cuda(x):
#     return x
#     # return x.cuda(non_blocking=True)

# def f1_score(y_true, y_pred, threshold=0.5):
#     return fbeta_score(y_true, y_pred, 1, threshold)


# def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
#     beta2 = beta**2

#     y_pred = torch.ge(y_pred.float(), threshold).float()
#     y_true = y_true.float()

#     true_positive = (y_pred * y_true).sum(dim=1)
#     precision = true_positive.div(y_pred.sum(dim=1).add(eps))
#     recall = true_positive.div(y_true.sum(dim=1).add(eps))

#     return torch.mean(
#         (precision*recall).
#         div(precision.mul(beta2) + recall + eps).
#         mul(1 + beta2))

# def train_one_epoch(model, train_loader, criterion, optimizer, steps_upd_logging = 250):
#     # model.train();
    
#     total_loss = 0.0
    
#     # train_tqdm = tqdm_notebook(train_loader)
    
#     for step, (features, targets) in enumerate(train_loader):
#         features, targets = cuda(features), cuda(targets)
        
#         # optimizer.zero_grad()
#         model.cleargrads()
        
#         logits = model(features)
        
#         loss = criterion(logits, targets)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
        
#         if (step + 1) % steps_upd_logging == 0:
#             logstr = f'Train loss on step {step + 1} was {round(total_loss / (step + 1), 5)}'
        
#     return total_loss / (step + 1)

# def validate(model, valid_loader, criterion, need_tqdm = False):
#     model.eval();
    
#     test_loss = 0.0
#     TH_TO_ACC = 0.5
    
#     true_ans_list = []
#     preds_cat = []
    
#     with torch.no_grad():
        
#         if need_tqdm:
#             valid_iterator = tqdm_notebook(valid_loader)
#         else:
#             valid_iterator = valid_loader
        
#         for step, (features, targets) in enumerate(valid_iterator):
#             features, targets = cuda(features), cuda(targets)

#             logits = model(features)
#             loss = criterion(logits, targets)

#             test_loss += loss.item()
#             true_ans_list.append(targets)
#             preds_cat.append(torch.sigmoid(logits))

#         all_true_ans = torch.cat(true_ans_list)
#         all_preds = torch.cat(preds_cat)
                
#         f1_eval = f1_score(all_true_ans, all_preds).item()

#     logstr = f'Mean val f1: {round(f1_eval, 5)}'
#     # kaggle_commit_logger(logstr)
#     return test_loss / (step + 1), f1_eval

# criterion = torch.nn.BCEWithLogitsLoss()
# # optimizer = torch.optim.Adam(lr=0.0005)
# # sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

# optimizer = Adam(learning_rate=0.01)

# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizer,
#               metrics=['accuracy'])

# # criterion = torch.nn.BCEWithLogitsLoss()
# # optimizer = torch.optim.Adam(model.weights, lr=0.0005)
# # sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

# TRAIN_LOGGING_EACH = 500

# train_losses = []
# valid_losses = []
# valid_f1s = []
# best_model_f1 = 0.0
# best_model = None
# best_model_ep = 0

# for epoch in range(1, N_EPOCHS + 1):
#     ep_logstr = f"Starting {epoch} epoch..."
#     # kaggle_commit_logger(ep_logstr)
#     tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, TRAIN_LOGGING_EACH)
#     train_losses.append(tr_loss)
#     tr_loss_logstr = f'Mean train loss: {round(tr_loss,5)}'
#     # kaggle_commit_logger(tr_loss_logstr)
    
#     valid_loss, valid_f1 = validate(model, test_loader, criterion)  
#     valid_losses.append(valid_loss)    
#     valid_f1s.append(valid_f1)       
#     val_loss_logstr = f'Mean valid loss: {round(valid_loss,5)}'
#     # kaggle_commit_logger(val_loss_logstr)
#     sheduler.step(valid_loss)
    
#     if valid_f1 >= best_model_f1:    
#         best_model = model        
#         best_model_f1 = valid_f1        
#         best_model_ep = epoch

# # x_train = x_train.reshape(60000, 784)
# # x_test = x_test.reshape(10000, 784)
# # x_train = x_train.astype('float32')
# # x_test = x_test.astype('float32')
# # x_train /= 255
# # x_test /= 255
# # print(x_train.shape[0], 'train samples')
# # print(x_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# # y_train = keras.utils.to_categorical(y_train, num_classes)
# # y_test = keras.utils.to_categorical(y_test, num_classes)

# # model.summary()


# # history = model.fit(x_train, y_train,
# #                     batch_size=batch_size,
# #                     epochs=epochs,
# #                     verbose=1,
# #                     validation_data=(x_test, y_test))

# # score = model.evaluate(x_test, y_test, verbose=0)
# # print('Test loss:', score[0])
# # print('Test accuracy:', score[1])
# # model.save('mnist_mlp_tanh.h5')

