import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import sklearn
from sklearn import *
from glob import glob
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
from scipy import stats
import time
start = time.time()
print(start)

submission = pd.read_csv('E:/dacon_data/kist/sample_submission.csv')
path = 'E:/dacon_data/kist01/kist/train/meta/'
path01 = 'E:/dacon_data/kist/test/meta/'
file_list = os.listdir(path)
file_list01 = os.listdir(path01)
file_list_py = [file for file in file_list if file.endswith('.csv')]
file_list_py01 = [file for file in file_list if file.endswith('.csv')]

file_name = []
file_name01 = []
for file in file_list:
    if file.count(".") == 1:
        name = file.split('.')[0]
        file_name.append(name)
    else:
        for k in range(len(file) -1, 0, -1):
            if file[k] == '.':
                file_name.append(file[:k])
                break
            
for file in file_list01:
    if file.count(".") == 1:
        name = file.split('.')[0]
        file_name01.append(name)
    else:
        for k in range(len(file) -1, 0, -1):
            if file[k] == '.':
                file_name01.append(file[:k])
                break

df = pd.DataFrame()
df01 = pd.DataFrame()
for i in file_list_py:
    data = pd.read_csv(path + i)
    df = pd.concat([df, data])

for i in file_list01:
    data = pd.read_csv(path01 + i)
    df01 = pd.concat([df01, data])

df = df.drop(columns = ['CO2관측치', 'EC관측치', '최근분무량', '블루 LED동작강도', '냉방부하', '난방온도', '청색광추정광량', '외부온도관측치'])
df01 = df01.drop(columns = ['CO2관측치', 'EC관측치', '최근분무량', '블루 LED동작강도', '냉방부하', '난방온도', '청색광추정광량', '외부온도관측치'])

Label_enc = sklearn.preprocessing.LabelEncoder()
df['시간'] = Label_enc.fit_transform(df['시간'])
df01['시간'] = Label_enc.fit_transform(df01['시간'])

for i in range(0, 1592):
    file_name[i] = df.iloc[0 + i * 1440 : 1440 + i * 1440, :]
for i in range(0, 460):
    file_name01[i] = df01.iloc[0 + i * 1440 : 1440 + i * 1440, :]

list01 = []
leaf = pd.DataFrame()
main_path = 'E:/dacon_data/kist/train/*'
main_folder = r'E:/dacon_data/kist/train/'
for item in os.listdir(main_folder):
    sub_folder = os.path.join(main_folder, item)
    if os.path.isdir(sub_folder):
        list01.append(sub_folder)

data_list = []
data_list01 = []
data_list02 = []
label_list = []
for i in list01:
    label_leaf = pd.read_csv(i + '/label.csv')
    leaf = pd.concat([leaf, label_leaf])

for i in range(0, 1592):
    data01 = file_name[i].to_numpy().reshape(1, 15840)
    data_list01.append(data01)

for i in range(0, 460):
    data01 = file_name01[i].to_numpy().reshape(1, 15840)
    data_list02.append(data01)

list01 = []
leaf = pd.DataFrame()
main_path = 'E:/dacon_data/kist/train/*'
main_folder = r'E:/dacon_data/kist/train/'
for item in os.listdir(main_folder):
    sub_folder = os.path.join(main_folder, item)
    if os.path.isdir(sub_folder):
        list01.append(sub_folder)

label_list = []
for i in list01:
    label_leaf = pd.read_csv(i + '/label.csv')
    leaf = pd.concat([leaf, label_leaf])

leaf_weight = leaf.iloc[:, 1]
weight = leaf_weight.to_numpy()
weight = weight.reshape(-1, 1)

# data processing 함수 
from tensorflow.keras.preprocessing.text import text_to_word_sequence

scaler = StandardScaler()

data_list01 = np.array(data_list01).reshape(1592, 15840)
data_list02 = np.array(data_list02).reshape(460, 15840)

train02 = pd.DataFrame(data_list01)
test02 = pd.DataFrame(data_list02)
train03 = pd.DataFrame(data_list01)
train02 = train02.fillna(0)
test02 = test02.fillna(0)
train03 = train03.fillna(0)

train02[:] = scaler.fit(train02[:])
test02[:] = scaler.transform(test02[:])
train03[:] = scaler.transform(train03[:])
weight01 = np.concatenate(weight).tolist()

x_train,x_test,y_train,y_test= sklearn.model_selection.train_test_split(train03,weight01,test_size=0.4, shuffle=True, random_state=0)

x_test,x_val,y_test,y_val= sklearn.model_selection.train_test_split(x_test, y_test,test_size=0.5, shuffle=False, random_state=0)



model = sklearn.neural_network.MLPRegressor(
hidden_layer_sizes = (1024, 64, 16, 8, 8, 8, 4),
activation = 'relu',
solver = 'adam',
learning_rate_init = 10 ** (-2.221947509012674),
max_iter = 1000,
batch_size = 512,
alpha = 10 ** (-2.3251),
warm_start = False,
random_state = 0 )
model.fit(x_train, y_train)
print(model.score(x_val, y_val))



print("time :", time.time() - start)

a = model.predict(test02)
submission.iloc[:, 1] = a
submission.to_csv('E:/dacon_data/kist/sample_submission_21.csv',index=False)
