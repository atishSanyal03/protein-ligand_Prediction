from keras.models import Sequential, Model
from keras.layers import Input, Conv3D,MaxPooling3D,Flatten,Dense, Dropout  , concatenate,Concatenate
from keras import optimizers, losses
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
#     train_y = np.append(train_y,np.zeros(1))
#     count = count + 1
#     print (count)
#
# np.save("x_full_train.npy",train_x)
# np.save("y_full_train.npy",train_y)
#
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
#     if count>=100:
#         break
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
        X_batch_1 = X_data[batch_size*counter:batch_size*(counter+1)][:,0]
        #print(X_batch_1.shape)
        X_batch_2 = X_data[batch_size*counter:batch_size*(counter+1)][:,1]
        y_batch = y_data[batch_size*counter:batch_size*(counter+1)]
        # print(y_batch.shape)
        counter += 1
        yield [X_batch_1,X_batch_2],y_batch

        if counter <= number_of_batches:
            counter = 0

def generatorPred(X_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0

    while 1:
        X_batch_1 = X_data[batch_size*counter:batch_size*(counter+1)][:,0]
        X_batch_2 = X_data[batch_size*counter:batch_size*(counter+1)][:,1]
        counter += 1

        yield [X_batch_1,X_batch_2]
        if counter <= number_of_batches:
            counter = 0

input1 = Input(shape=(11,11,11,1))
input2 = Input(shape=(11,11,11,1))
#model = Sequential()

conv1= Conv3D(filters=32,kernel_size=3,padding='Same',data_format='channels_last',activation='linear')(input1)
#model.add(MaxPooling3D(pool_size=3))
#model.add(Dropout(0.5))

conv2 = Conv3D(filters=64,kernel_size=3,padding='Same',data_format='channels_last',activation='linear')(conv1)
# model.add(MaxPooling3D())
# model.add(Dropout(0.5))

# model.add(Conv3D(filters=64,kernel_size=3,padding='Same',data_format='channels_last',activation='linear'))
pool1 = MaxPooling3D(pool_size=3)(conv2)
dropout1 = (Dropout(0.5))(pool1)
flat1 = Flatten()(dropout1)

conv3= Conv3D(filters=32,kernel_size=3,padding='Same',data_format='channels_last',activation='linear')(input2)
#model.add(MaxPooling3D(pool_size=3))
#model.add(Dropout(0.5))

conv4 = Conv3D(filters=64,kernel_size=3,padding='Same',data_format='channels_last',activation='linear')(conv3)
# model.add(MaxPooling3D())
# model.add(Dropout(0.5))

# model.add(Conv3D(filters=64,kernel_size=3,padding='Same',data_format='channels_last',activation='linear'))
pool2 = MaxPooling3D(pool_size=3)(conv3)
dropout2 = (Dropout(0.5))(pool2)
flat2 = Flatten()(dropout2)

merge=concatenate([flat1,flat2])
#model.add(concatenate([branch1,branch2],axis=0))
#model.add(result)
# model.add(Dense(1024,activation='linear'))
hidden1 = (Dense(512,activation='linear'))(merge)
hidden2 = (Dense(32,activation='linear'))(hidden1)
output = (Dense(2,activation='softmax'))(hidden2)

# sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
adamax = optimizers.Adamax(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=0, decay=0.0)
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=0, decay=0.0)

imbalance_weight = {0: 1.,1:20}

model = Model(inputs = [input1,input2],outputs = output)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=adamax)

#for i in range(5):
model.fit_generator(generator(train_x, train_y, batch_size=32),shuffle=True,epochs=1, verbose=1,steps_per_epoch=train_x.shape[0]/32,class_weight=imbalance_weight)
y_pred=model.predict_generator(generatorPred(test_x,32),steps=test_x.shape[0]/32)
#print (loss,"    ",acc)
print (y_pred[:1],test_y)
print (y_pred.shape,test_y.shape)
confVal=confusion_matrix(np.argmax(test_y[:1],axis=1),np.argmax(y_pred[:1],axis=1))
print (confVal)

model.save("first_model.h5")