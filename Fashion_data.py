import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data() # load dataset from  keras

y_train[0] #check
plt.imshow(X_train[0])

#X_train.shape --> (60000,28,28)
#X_train[0].shape --> (28,28)
#------------After expanding dimention----------------------------
#X_train.shape --> (60000,28,28,1)
#X_train[0].shape --> (28,28,1)

X_train = np.expand_dims(X_train, -1) #changing dimension.because, conv requires 4D
X_test = np.expand_dims(X_test, -1)

X_train = X_train/255.0
X_test = X_test/255.0

#creating the model
model = keras.models.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='valid',activation= 'relu', input_shape=[28,28,1]),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=10, activation='softmax')
])


model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

model.predict(X_test)[1].round(2)   #specific element prediction
y_test[1] #checking the specific prediction corrected or not

y_pred = model.predict(X_test).round(2)   #whole dataset prediction
y_pred

model.evaluate(X_test, y_test)   #overall evaluation on test data

#For decoration of Confusion Metrix
class_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
'''
0 => T-shirt/top
1 => Trouser
2 => Pullover
3 => Dress
4 => Coat
5 => Sandal
6 => Shirt
7 => Sneaker
8 => Bag
9 => Ankle boot '''


#To show the confusion metrix
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(16,9))
y_pred_labels = [ np.argmax(label) for label in y_pred ]
cm = confusion_matrix(y_test, y_pred_labels)

# show cm
sns.heatmap(cm, annot=True, fmt='d',xticklabels=class_labels, yticklabels=class_labels)


see the recall,precision,accuracy etc as report
from sklearn.metrics import classification_report
cr= classification_report(y_test, y_pred_labels, target_names=class_labels)
print(cr)
