from keras.models import Sequential
from keras.layers import Input, Conv3D,MaxPooling3D,Flatten,Dense, Dropout, BatchNormalization
from keras import optimizers, losses,regularizers
from keras.initializers import glorot_normal
from keras.utils.np_utils import to_categorical
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
#os.environ['CUDA_VISIBLE_DEVICES']=" "
# train_x = np.load("x_positive_train.npy")
# train_y = np.ones(train_x.shape[0])
#
# arr = os.listdir("./negative_train")
# count = 0
# for x in arr:
#     # print(train_x.shape)
#     # print(train_y.shape)
#     train_x = np.append(train_x,np.load("./negative_train/" + x),axis=0)
#     train_y = np.append(train_y,np.zeros(2))
#     count = count + 1
#     print (count)
#
# np.save("x_full_train.npy",train_x)
# np.save("y_full_train.npy",train_y)

train_x = np.load("x_full_train.npy")
train_y = np.load("y_full_train.npy")
train_y = to_categorical(train_y,num_classes=2)
# test_x=np.load("x_positive_test.npy")
# test_y=np.ones(test_x.shape[0])
# arr = os.listdir("./negative_test")
# count = 0
# for x in arr:
#     # print(train_x.shape)
#     # print(train_y.shape)
#     test_x = np.append(test_x,np.load("./negative_test/" + x),axis=0)
#     test_y = np.append(test_y,np.zeros(200))
#     count = count + 1
#
#     print (count)
#
# np.save("x_full_test.npy",test_x)
# np.save("y_full_test.npy",test_y)

test_x = np.load("x_full_test.npy")
test_y = np.load("y_full_test.npy")
# c=0
# o=0
# for y in test_y:
#     if y==[0]:
#         print (y)
#         c+=1
#     else:
#         o+=1
# print ("C===",c)
# print ("O===",o)

test_y = to_categorical(test_y,num_classes=2)

train_x,train_y=shuffle(train_x,train_y,random_state=0)
test_x,test_y=shuffle(test_x,test_y,random_state=0)

def generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]

    number_of_batches = samples_per_epoch/batch_size

    counter=0



    while 1:
        X_batch = X_data[batch_size*counter:batch_size*(counter+1)]

        y_batch = y_data[batch_size*counter:batch_size*(counter+1)]

        counter += 1

        yield X_batch,y_batch
        if counter <= number_of_batches:

            counter = 0

def generatorPred(X_data, batch_size):
    samples_per_epoch = X_data.shape[0]

    number_of_batches = samples_per_epoch/batch_size

    counter=0



    while 1:
        X_batch = X_data[batch_size*counter:batch_size*(counter+1)]



        counter += 1

        yield X_batch
        if counter <= number_of_batches:

            counter = 0


model = Sequential()

model.add(Conv3D(filters=32,kernel_size=3,padding='Same',data_format='channels_last',activation='relu', use_bias=True,input_shape=(11,11,11,2)))
#model.add(MaxPooling3D(pool_size=3))
#model.add(Dropout(0.5))


model.add(Conv3D(filters=64,kernel_size=3,padding='Same',data_format='channels_last', use_bias=True,activation='relu',kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))

model.add(Conv3D(filters=128,kernel_size=3,padding='Same',data_format='channels_last', use_bias=True,activation='relu',kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))

model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
# model.add(MaxPooling3D())
# model.add(Dropout(0.5))

# model.add(Conv3D(filters=64,kernel_size=3,padding='Same',data_format='channels_last',activation='linear'))
model.add(MaxPooling3D(pool_size=3))
model.add(Dropout(0.5))
model.add(Flatten())
# model.add(Dense(1024,activation='linear'))
model.add(Dense(512,activation='linear'))
model.add(Dense(32,activation='linear'))
model.add(Dense(2,activation='softmax'))

# sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
adamax = optimizers.Adamax(lr=0.0006, beta_1=0.9, beta_2=0.999, epsilon=0, decay=0.0)
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=0, decay=0.0)
adam = optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=0, decay=0.0)
imbalance_weight = {0: 1,1:2}

model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=adamax)
#for i in range(5):
train_x=train_x
train_y=train_y
print (train_x.shape,train_y.shape)

model.fit_generator(generator(train_x, train_y, batch_size=128),shuffle=False,epochs=25, verbose=1,steps_per_epoch=train_x.shape[0]/128,class_weight=imbalance_weight)
y_pred=model.predict_generator(generatorPred(test_x,128),steps=test_x.shape[0]/128)
#print (loss,"    ",acc)
print (y_pred,test_y)
print (y_pred[:120600].shape,test_y.shape)
confVal=confusion_matrix(np.argmax(test_y,axis=1),np.argmax(y_pred[:120600],axis=1))
print (confVal)

model.save("first_model.h5")