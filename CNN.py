import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

cnn=Sequential()
cnn.add(Conv2D(40,(4,4),input_shape=(64,64,3)))
cnn.add(MaxPooling2D(2,2))
cnn.add(Conv2D(40,(4,4)))
cnn.add(MaxPooling2D(2,2))
cnn.add(Flatten())
cnn.add(Dense(64,activation="relu"))
cnn.add(Dense(32,activation="relu"))
cnn.add(Dense(16,activation="relu"))
cnn.add(Dense(8,activation="relu"))
cnn.add(Dense(4,activation="relu"))
cnn.add(Dense(1,activation="sigmoid"))

cnn.compile(optimizer="adam",loss="binary_crossentropy")

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        r"C:\\Users\\itsde\\OneDrive\\Desktop\\deep learning\\cnn dataset\\training_set\\training_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        r"C:\\Users\\itsde\\OneDrive\\Desktop\\deep learning\\cnn dataset\\test_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

cnn.fit(train_generator,epochs=20,validation_data=test_generator, )

img=image.load_img("C:\\Users\\itsde\\OneDrive\\Desktop\\deep learning\\cnn dataset\\test_set\\test_set\\cats\\cat.4048.jpg",target_size=(64,64))
img=image.img_to_array(img)
img=np.expand_dims(img,axis=0)
pre=cnn.predict(img)
print(pre)
if pre[0][0]<0.5:
    print("cat")
else:
    print("dog")
print(train_generator.class_indices)
print(train_generator.samples)
