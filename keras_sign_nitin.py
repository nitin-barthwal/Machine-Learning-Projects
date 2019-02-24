# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:27:13 2018

@author: Nitin
"""


# Assignment 2
# Demonstration of Keras and Bokeh
# 
# Submitted by: Nitin Barthwal
# 

# Keras Basic Introduction
# 
# Keras is a Python library that provides, in a simple way, the creation of a wide range of Deep Learning models using as backend other libraries such as TensorFlow, Theano or CNTK. 
# 
# Bokeh Basic Introduction
# 
# 

# We will install Tensorflow as backend libraries for Keras.
# By default, Keras is configured to use Tensorflow as the backend since it is the most popular choice. 

# The basic flow of the report will be:-
# 1. Install and import Libraries
# 2. Download Data/CSV file
# 3. Load the data.
# 4. Split training data into train and validation data
# 5. Reshape data 
# 6. Preprocess & Standardize data.
# 7. Dispay test data using Bokeh
# 8. Define Model.
# 9. Compile Model.
# 10. Train Model.
# 11. Evaluate Model.
# 12. Tie It All Together.

# # Install / Import  Libraries
# At the Conda Prompt write the command
# conda install keras
# conda install bokeh

# To check version 
# import keras
# keras.__version__


#1
import tensorflow as tf
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D
from bokeh.plotting import figure, output_file, show ,gridplot
#from bokeh.io import export_png 
from sklearn.model_selection import train_test_split
import numpy as np

from keras.callbacks import ModelCheckpoint


#2
# Download Data/CSV file provided with the code file.

#3
dataset_train = np.loadtxt('sign_language_train.csv', delimiter=',',dtype=int ,skiprows=1)
x_train = dataset_train[:,1:]
y_train = dataset_train[:,0]
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape) 

dataset_test = np.loadtxt('sign_language_test.csv', delimiter=',',dtype=int ,skiprows=1)
x_test = dataset_test[:,1:]
y_test = dataset_test[:,0]
print("x_test shape:", x_test.shape, "y_test shape:", y_test.shape) 



#4
# # Split the data into train/validation/test data sets 
# Training data - used for training the model
# Validation data - used for tuning the hyperparameters and evaluate the models
# Test data - used to test the model after the model has gone through initial vetting by the validation set.


x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=15)

#5
# # Reshape data according to picture size
# Here the image is stored as 28*28 pixel size

x_train=np.reshape(x_train, (x_train.shape[0], 28,28))
x_test=np.reshape(x_test, (x_test.shape[0], 28,28))
x_valid=np.reshape(x_valid, (x_valid.shape[0], 28,28))

print("After Reshape x_train shape:", x_train.shape, "y_train shape:", y_train.shape) 
print("After Reshape x_test shape:", x_test.shape, "y_test shape:", y_test.shape) 
print("After Reshape x_valid shape:", x_valid.shape, "y_valid shape:",y_valid.shape) 


#7
# # Display an image using Bokeh
# - Bokeh is a data visualization library for python
# - Bokeh is built for the web
# - Creates dynamic and interactive plots/figures that are driven by data
# - Filling in the parameters to use the `image` renderer to display the an image with color mapped with the palette 'Spectral11'
# - p1.image(image=               # image data
#          x=                   # lower left x coord
#          y=                   # lower left y coord
#          dw=                  # *data space* width of image
#          dh=                  # *data space* height of image
#          palette=             # palette name
# 
# A Figure is the grouping of all the elements (i.e. the plot) and Glyphs are the basic visual markers that Bokeh can display. The most basic Glyph is a line. There are also many other Glyphs, such as circles, bars, wedges, etc. The great thing is that you can stack multiple Glyphs on a single Figure.



# # Data normalization
# Normalize the data dimensions so that they are of approximately the same scale.

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# # Converts a class vector (integers) to binary class matrix.
# 
# E.g. for use with categorical_crossentropy.
# 
# Arguments
# 
# y: class vector to be converted into a matrix (integers from 0 to num_classes).
# num_classes: total number of classes.
# dtype: The data type expected by the input, as a string (float32, float64, int32...)


# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

y_train = tf.keras.utils.to_categorical(y_train, 25)
y_valid = tf.keras.utils.to_categorical(y_valid, 25)
y_test = tf.keras.utils.to_categorical(y_test, 25)


# Print training set shape
print("New x_train shape:", x_train.shape, "New y_train shape:", y_train.shape)


# # Define Model in Keras : Sequential
model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, 
                                 kernel_size=2, 
                                 padding='same', 
                                 activation='relu', #
                                 input_shape=(28,28,1))) 


# ## Activation Function
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

# ### Arguments for Dropout Layer
# Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.
# The effect is that the network becomes less sensitive to the specific weights of neurons. This in turn results in a network that is capable of better generalization and is less likely to overfit the training data.
# 
# - rate: float between 0 and 1. Fraction of the input units to drop.

model.add(tf.keras.layers.Dropout(0.3))

# ###  Flatten Layer
# 
# It Flattens the input. Does not affect the batch size.
# 
# -data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. The purpose of this argument is to preserve weight ordering when switching a model from one data format to another. channels_last corresponds to inputs with shape  (batch, ..., channels) while channels_first corresponds to inputs with shape (batch, channels, ...). It defaults to the image_data_format value found in your Keras config file at  ~/.keras/keras.json.
#     - If you never set it, then it will be "channels_last".
#     -  Dropout is only applied during training.
# 

model.add(tf.keras.layers.Flatten())

# ###  Dense Layer 
# A dense layer is a classic fully connected neural network layer : each input node is connected to each output node.
# Arguments
# - units: Positive integer, dimensionality of the output space.
# - activation: Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).

model.add(tf.keras.layers.Dense(256, activation='relu'))
# Remove 50% of the Neurons : technique used to overcome overfitting
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(25, activation='softmax'))

# Take a look at the model summary
model.summary()


# # Compile Model
# Configure the learning process with compile() API before training the model. It receives three arguments:

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


#ModelCheckpoint Saves the model after every epoch.
# The weight file is created in the local directory.
# # weights.best.hdf5
# For example: if filepath is weights.{epoch:02d}-{val_loss:.2f}.hdf5, then the model checkpoints will be saved with the epoch number and the validation loss in the filename.
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)

epochs_count =2
history = model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=epochs_count,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer])


# Here we are displaying the 
colors_list = ['blue', 'red']
legends_list = ['Test Accuracy', 'Validation Accuracy']
legends_list2 = ['Test Loss', 'Validation Loss']
ys=[history.history['acc'], history.history['val_acc']]
xs=[ range(0,epochs_count), range(0,epochs_count)]
acc = figure(plot_width=400, plot_height=400 ,title='Accuracy Plot')
for (colr, leg, x, y ) in zip(colors_list, legends_list, xs, ys):
    my_plot = acc.line(x, y, color= colr, legend= leg)
acc.yaxis.axis_label = "Accuracy"
acc.xaxis.axis_label = "Epochs"

ys_loss=[history.history['loss'], history.history['val_loss']]
xs_loss=[ range(0,epochs_count), range(0,epochs_count)]
val = figure(plot_width=400, plot_height=400 ,title='Loss Plot')
for (colr, leg, x, y ) in zip(colors_list, legends_list2, xs_loss, ys_loss):
    my_plot = val.line(x, y, color= colr, legend= leg)
val.yaxis.axis_label = "Loss"
val.xaxis.axis_label = "Epochs"


# Using the set checkpoint we load the weights with the best validation accuracy
# The model is then used to make predictions on the entire dataset.
model.load_weights('model.weights.best.hdf5')


# # Test accuracy
# Test the accuracy of the model on test data

score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])


y_hat = model.predict(x_test)

# # bokeh

c= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
#Displaying the Scatter Plot
f = figure(plot_width=400, plot_height=400,title= "Scatter Plot of first 100 values ")
a=list()
for i in range(0,100):
    a.append (np.argmax(y_test[i]))
f.circle(a, range(0,100), size=1.5, color="navy", alpha=1)
f.xaxis.ticker = c
f.yaxis.axis_label = "Indexes"
f.xaxis.axis_label = "Classes ( Fashion Items )"

# Displaying the images at various indexes choosen randomly
index=4        
p  = figure( title='Predicted Image at index 4',plot_width=300, plot_height=300)
# must give a vector of image data for image parameter
#need to flip data using np.flipud otherwise image is displaying upsdide down
 
p.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=2, dh=2, palette="Spectral11")
p.outline_line_width = 9
p.outline_line_alpha = 0.5

predict_index = np.argmax(y_hat[index])
true_index = np.argmax(y_test[index])
p.outline_line_color = ("green" if predict_index == true_index else "red")
p.background_fill_color = "beige"

index=5
q  = figure( title='Predicted Image  at index 5',plot_width=300, plot_height=300)
q.image(image=[np.flipud(x_test[index+1])], x=0, y=0, dw=2, dh=2, palette="Spectral11")
q.outline_line_width = 9
q.outline_line_alpha = 0.5
predict_index = np.argmax(y_hat[index])
true_index = np.argmax(y_test[index])
q.outline_line_color = ("green" if predict_index == true_index+1 else "red")
q.background_fill_color = "beige"

index=11
r  = figure( title='Predicted Image at index 11',plot_width=300, plot_height=300)
r.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=2, dh=2, palette="Spectral11")
r.outline_line_width = 9
r.outline_line_alpha = 0.5
predict_index = np.argmax(y_hat[index])
true_index = np.argmax(y_test[index])
r.outline_line_color = ("green" if predict_index == true_index else "red")
r.background_fill_color = "beige"

index=16
s  = figure( title='Predicted Image at index 16',plot_width=300, plot_height=300)
s.image(image=[np.flipud(x_test[index+1])], x=0, y=0, dw=2, dh=2, palette="Spectral11")
s.outline_line_width = 9
s.outline_line_alpha = 0.5
predict_index = np.argmax(y_hat[index])
true_index = np.argmax(y_test[index])
s.outline_line_color = ("green" if predict_index == true_index + 1 else "red")

s.background_fill_color = "beige"


# Original Images in test set
a = figure(title='Original Image at index 4',plot_width=300, plot_height=300)
index=4
# must give a vector of image data for image parameter
#need t0 flip data using np.flipud otherwise image is displaying upsdide down
a.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=14, dh=14, palette="Spectral11")

index=5
b = figure(title='Original Image at index 5',plot_width=300, plot_height=300)
b.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=14, dh=14, palette="Spectral11")

index=11
c = figure(title='Original Image at index 11',plot_width=300, plot_height=300)
c.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=14, dh=14, palette="Spectral11")

index=16
d = figure(title='Original Image at index 16',plot_width=300, plot_height=300)
d.image(image=[np.flipud(x_test[index])], x=0, y=0, dw=14, dh=14, palette="Spectral11")


# using grid plot to show all of the bokeh visualisation in one html file ( same Plot )
h = gridplot([[p, q, r,s],[a, b, c,d], [acc, val, f]] )
output_file('graph.html',)
# Show 2 of the images from the training dataset in a grid
show(h)

