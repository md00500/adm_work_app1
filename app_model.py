## 2024/04/09 ドキュメント画像を変更（2_doc_img)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
import pandas as pd
# ファイルパス
path_gen = os.listdir('./dataset/0_gen_pic/') #path_male -> path_gen
path_doc = os.listdir('./dataset/2_doc_pic/') #path_female -> path_doc
#print(path_doc)
img_gen = []
img_doc = []

np.random.seed(0)
path_tmp = np.array(path_gen)
rand_index = np.random.permutation(np.arange(len(path_tmp)))
print("path_gen",len(path_gen),"path_doc",len(path_doc))
#print(rand_index)
path_gen = path_tmp[rand_index]

for i in range(100):
#for i in range(len(path_gen)):
    img = cv2.imread('./dataset/0_gen_pic/' + path_gen[i])
    if i == 0:print(path_gen[i],img.shape)
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = cv2.resize(img, (50,50))
    if i == 0:print(path_gen[i],img.shape)
    img_gen.append(img)
#print(path_gen)
for i in range(100):
#for i in range(len(path_doc)):
    img = cv2.imread('./dataset/2_doc_pic/' + path_doc[i])
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = cv2.resize(img, (50,50))
    img_doc.append(img)

X = np.array(img_gen + img_doc)
y =  np.array([0]*len(img_gen) + [1]*len(img_doc))
# 一般画像ファイル
# ドキュメント画像ファイル
print(X.shape)
print(y.shape)
print()
rand_index = np.random.permutation(np.arange(len(X)))
X = X[rand_index]
y = y[rand_index]
print("len X =",len(X))
#print(rand_index)
# データの分割
X_train = X[:int(len(X)*0.8)]
y_train = y[:int(len(y)*0.8)]
X_test = X[int(len(X)*0.8):]
y_test = y[int(len(y)*0.8):]
print(X_train.shape,y_train.shape)

input_tensor = Input(shape=(50, 50, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
print(vgg16.output)
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
#top_model.add(Dense(1, activation='softmax'))
top_model.add(Dense(1, activation='sigmoid'))
# モデルの連結
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
# 19層目までの重みをfor文を用いて固定する
for layer in model.layers[:19]:
    layer.trainable = False

#model.compile(loss='sparse_categorical_crossentropy',
#              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#              metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',
#              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#              metrics=['accuracy'])
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
              #optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=10)
#history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=50)

# 精度の評価
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# 学習過程
print(history.history)
# Pandas 形式
result = pd.DataFrame(history.history)
print(result.head())
# 目的関数の値
#result[['loss', 'val_loss']].plot();
# 正解率
#result[['accuracy', 'val_accuracy']].plot();

#resultsディレクトリを作成
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
# 重みを保存
model.save(os.path.join(result_dir, 'model.keras'))
