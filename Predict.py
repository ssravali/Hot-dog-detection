print('Loading Libraries.....')
import numpy as np
import pandas
import os
import time
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

print("Loaded.......")

#Getting file name
cwd = os.getcwd()
f = str(input("Enter the location of the image file"))
#check if file exists
if not (os.path.exists(f)):
    raise ValueError("File name doesn't exist")
start_time = time.time()
#loafing model
print("Loading model.....................")
model = load_model('my_model.h5')
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#Loading Image
test_image = img_to_array(load_img(f, target_size=(100, 100))) / 255.0
test_image = np.expand_dims(test_image, axis=0)

#Predicting
print("Predicting......................")
prediction = model.predict(test_image)[0][0]
print("The prediction for " + f + " is:")
if prediction < 0.5:
    print("The image is of a hotdog! Probability: {0:.2%}\n".format(1 - prediction))
else:
    print("The image is not of a hotdog. Probability: {0:.2%}\n".format(prediction))
print("Request completed in: ",(time.time() - start_time))    
