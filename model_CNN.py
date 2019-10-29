#libraries to import
import cv2
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pickle
import zipfile
from keras.callbacks import Callback, TensorBoard
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.optimizers import Adam, rmsprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
print("Loading libraries..........")

class Metrics(Callback):
    #To monitor f1 Values of validation set
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict,average='micro')
        _val_recall = recall_score(val_targ, val_predict,average='micro')
        _val_precision = precision_score(val_targ, val_predict,average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("— val_f1:",_val_precision,'precision: ',_val_precision,'recall: ',_val_recall)
        return()
    
#loading data
def open_zip(folder):
    folder = str(os.getcwd() + '/' + folder)
    with zipfile.ZipFile(folder,'r') as zip_ref:
        zip_ref.extractall(str(os.getcwd()))

def load_images_from_folder2(folder):
    #loads and converts images from a folder
    #resizing each image to 100
    images = []
    for filename in os.listdir(folder):
        img = img_to_array(load_img(os.path.join(folder,filename), target_size=(100, 100))) / 255.0
        if img is not None:
            images.append(img)
    return images


cwd = os.getcwd()
print("Unzipping data.................")
#Opening the zip folder of images
open_zip('dataset.zip')
print("Loading data...................")
#loading data
train_x_hd = np.asarray(load_images_from_folder2(str(cwd + '/dataset/hd')))
train_x_nhd = np.asarray(load_images_from_folder2(str(cwd + '/dataset/nhd')))
Y_hd = np.array([0]*len(train_x_hd))
Y_nhd = np.array([1]*len(train_x_nhd))
Y_hd = Y_hd.reshape(-1, 1)
Y_nhd = Y_nhd.reshape(-1, 1)
print("splitting data.....................")
#splitting data into training,testing and validation sets for Hotdog images
x_train1_hd, x_test_hd, y_train1_hd, y_test_hd = train_test_split(train_x_hd, Y_hd, test_size=0.2, random_state=42)
x_train_hd, x_val_hd, y_train_hd, y_val_hd = train_test_split(x_train1_hd[:410], y_train1_hd[:410], test_size=0.2, random_state=42)
#splitting data into training,testing and validation sets for non - Hotdog images
x_train1_nhd, x_test_nhd, y_train1_nhd, y_test_nhd = train_test_split(train_x_nhd, Y_nhd, test_size=0.2, random_state=1)
x_train_nhd, x_val_nhd, y_train_nhd, y_val_nhd = train_test_split(x_train1_nhd[410:], y_train1_nhd[410:], test_size=0.2, random_state=42)

#Creating train, test and validation final sets
x_train = np.vstack((x_train_hd, x_train_nhd[:450]))
y_train = np.vstack((y_train_hd, y_train_nhd[:450]))
x_test = np.vstack((x_test_hd, x_test_nhd))
y_test = np.vstack((y_test_hd, y_test_nhd))
x_val = np.vstack((x_val_hd, x_val_nhd))
y_val = np.vstack((y_val_hd, y_val_nhd))

#Agumenting Hot dog images to add extra 2000 images of hotdogs
#This will uniform the dataset from hotdogs perspectives
print("Augumenting Data...............")
ag_x_train,ag_y_train = Augumentation(train_x_hd,2000)
new_x_train = np.vstack((x_train,ag_x_train))
new_y_train = np.vstack((y_train,ag_y_train))
print("Agumented new datset size: ",new_x_train.shape,new_y_train.shape)

print("Creating Model..........")
#Model
image_size = new_x_train.shape[1]
batch_size = 16  
metrics = Metrics()
inp = Input(shape=(image_size,image_size,3))
x = Conv2D(32, (3,3), activation='relu')(inp)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu',padding='same')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu',padding='same')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inp,x)
model.summary()
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

print("Fitting the model.......")
#Data generator to increase dataset size and add differenct perspectives
datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.3, height_shift_range=0.2, shear_range=0.2, zoom_range=0.3, horizontal_flip=True, vertical_flip=True)
es = EarlyStopping(monitor='val_loss', verbose=1, patience=100)
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0] // batch_size, epochs=500,validation_data=[x_val,y_val],verbose = 1,shuffle=True, callbacks=[metrics,es])
#history = model.fit(x = new_x_train,y = new_y_train, epochs = 100, verbose = 1, validation_data=[new_x_val,new_y_val],shuffle=True,batch_size=batch_size, callbacks=[metrics,tensorboard,es])
model.save('my_model.h5')
score = model.evaluate(x=x_test,y = y_test, verbose = 1)
print("The network has a Loss: ",score[0],"acc: ",score[1])

#import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
