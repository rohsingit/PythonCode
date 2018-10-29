#pip install --upgrade tensorflow
import tensorflow as tf
tf.__version__

mnist = tf.keras.datasets.mnist   # 28 x 28 images of handwritten digits
(x_train, y_train), (x_test, y_test) = mnist.load_data() 


import matplotlib.pylot as plt
plt.inshow(x_train[0], cmap = plt.cm.binary)
plt.show()
print(x_train[0]) #basically matrix of values aka tensor 0 to 255

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
model = tf.keras.models.Sequential()    #one of two models
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))    #128 neurons, rectified linear activation.. can be sigmoid, stepper etc.
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))    #adding 2nd layer
model.add(tf.keras.layers.Dense(128, activation = tf.nn.softmax))   #softmax for probability distribution


model.compile(optimizer = 'adam',     #could be stochastic gradient descent.. 10 or so available
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 3)
            
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save("tf_reader.model")
new_model = tf.keras.models.load_model("tf_reader.model")
predictions = new_model.predict([x_test])

import numpy as np
print(np.argmax(predictions[0]))      #comverts the prediction to a number

plt.inshow(x_test[0])                 #check real picture
plt.show()
