# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:14:48 2021

@author: abhij
"""

from keras.models import Sequential
from keras.layers import Conv2D, SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

model= Sequential()

model.add(Conv2D(64,(3,3),input_shape=(200,200,3),activation='relu',padding='same'))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(SeparableConv2D(128,(3,3), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(128,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(SeparableConv2D(256,(3,3), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(256,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(SeparableConv2D(256,(3,3), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(256,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(SeparableConv2D(512,(3,3), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(512,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.7))

model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=5, activation='softmax'))
model.summary()

opt = Adam(learning_rate=0.0001, decay=1e-5)
es = EarlyStopping(patience=5,min_delta= .05, monitor="val_accuracy")
cp = ModelCheckpoint(filepath='best_model', save_best_only=True, save_weights_only=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=opt)

batch_size = 16 
training_images = 'training_set'
testing_images = 'testing_set'

TrainDatagen = ImageDataGenerator(
        preprocessing_function= preprocess_input,
        horizontal_flip = True)

TestDatagen = ImageDataGenerator(
    preprocessing_function= preprocess_input)

train_data = TrainDatagen.flow_from_directory(
    training_images,
    target_size = (200,200),
    batch_size =batch_size,
    class_mode = 'categorical')

test_data = TestDatagen.flow_from_directory(
    testing_images,
    target_size = (200,200),
    batch_size = batch_size,
    class_mode = 'categorical')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
history = model.fit(
    train_data,              
    steps_per_epoch = train_data.samples//batch_size,
    validation_data = test_data,
    validation_steps = test_data.samples//batch_size,
    epochs = 20,
    callbacks=[es,cp])

model.save(filepath='my_model')



import keras
new_model=keras.models.load_model('my_model')

predpath='prediction_set'
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

pred_data= ImageDataGenerator( preprocessing_function= preprocess_input)

pred=pred_data.flow_from_directory(
    predpath,
    target_size = (200,200),
     batch_size = 2,
    class_mode = 'categorical')
import numpy as np
prediction=new_model.predict(pred)
ans= np.argmax(prediction)

if ans==0:
    print("Heart")
if ans==1:
    print("Oblong")
if ans==2:
    print("Oval")
if ans==3:
    print("Round")
if ans==4:
    print("Square")

