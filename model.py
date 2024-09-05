import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import shutil
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# mixxing fake and real dataset
from Dataset_Folder import copyfolder
fake = 'D:\Deep_Fake\DataSet\Train\Fake'
real = 'D:\Deep_Fake\DataSet\Train\Real'
new =  'D:\Deep_Fake\DataSet\Train_Mix'

# copyfolder(fake, real, new)

from fake_real_count import count_file
fake_img,real_img= count_file(new)
print('Fake Images : ', fake_img)
print('real Images : ', real_img)

from Set_Labels import set_label
labels = set_label(new)
print(len(labels))


path = 'D:/Deep_Fake/DataSet/Train_Mix/'
files = os.listdir('D:/Deep_Fake/DataSet/Train_Mix')
fake_real_img = []
labels2 = []
cnt =0

for file in files :
    cnt += 1
    if cnt%100==0:
        labels2.append(labels[cnt])
        name = path+file
        img = Image.open(name)
        img = np.array(img)
        fake_real_img.append(img)

        
        if cnt%1000==0:
            print('processed data : ', cnt )

label_map = {'fake': 0, 'real': 1}
labels_numeric = [label_map[label] for label in labels2]

X = np.asarray(fake_real_img)
Y = np.asarray(labels_numeric)

print('Shape of input : ',X.shape)
print('Shape of output : ',Y.shape)




# Normalize the image data
X = X.astype('float32') / 255.0

# Convert labels to categorical
Y = to_categorical(Y)

# Split the data into training, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Load the VGG16 model without the top dense layers
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Freeze the convolutional base
conv_base.trainable = False

# Build the model
model = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=2e-5),
              metrics=['accuracy'])

# Print model summary
print(model.summary())

# Train the model
history = model.fit(
    X_train, Y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, Y_val)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)

# Plot the training and validation accuracy and loss
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# Predictive system
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    label_map = {0: 'fake', 1: 'real'}
    predicted_label = np.argmax(prediction, axis=1)[0]
    return label_map[predicted_label]

# Example usage:
image_path = 'rimage1.jpg'
prediction = predict_image(image_path)
print(f'The predicted label for the image is: {prediction}')




# Save the model as HDF5
model.save('fake_real_model3.h5', include_optimizer=False)







