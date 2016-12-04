
# coding: utf-8

# In[1]:


# import keras modules
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dropout, Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

# sklearn libraries for preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

# Numeric Libraries
import numpy as np

# Image Libraries
import cv2
from imutils import paths

# System Libraries
import os
import csv


# In[2]:

# Extract feature vectors from images using fitler 'size'      
def image_to_feature_vector(image, size=(32, 32)):
    ''' Expects the image to be a RGB image and then resize the given image using a filter size.
    Transforms the image to suppport theano implementation.
    Ensures that every image is of fixed size '''
    im = cv2.resize(image, size).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose(2, 0, 1)
    return im


# In[4]:

# Loads the data and returns the list of image vectors
def load_data(data_path='../data/train', labels_path='../data/trainLabels.csv'):
    imagePaths = list(paths.list_images(data_path))
    imagePaths.sort()
    shape = len(imagePaths)
    
    labels_dict = load_labels()
    data = np.zeros((shape, 3, 32, 32))
    labels = []
    for index in range(shape):
        # Read Image Id
        imageId = int(imagePaths[index].split(os.path.sep)[-1].split(".")[0])
        im = cv2.imread(imagePaths[index])
        
        # Extract feature vectors from images using 'image_to_feature_vector'
        features = image_to_feature_vector(im)
        data[index] = features
        labels.append(labels_dict[imageId])
    # Normalize the data (by highest intesity - '255.0')
    data = np.array(data)/ 255.0    
    return data, labels


# In[5]:

def load_labels(data_path='../data/trainLabels.csv'):
    fp = open(data_path,'rb')
    labels = {}
    data = csv.reader(fp, delimiter=',')
    next(data)
    for row in data:
        labels[int(row[0])] = row[1]
    return labels


# In[7]:

class_to_labels =['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 
                  'horse', 'ship', 'truck']
def write_output(filename, imageIds, test_labels):
    fp = open(filename, 'wb')
    fp.write('id,label\n')
    for (imageId, label) in zip(imageIds, test_labels):
        fp.write('{0},{1}\n'.format(imageId,class_to_labels[label]))
    fp.close()


# In[8]:

def load_test_data(data_path='../data/test'):
    imagePaths = list(paths.list_images(data_path))
    imagePaths.sort()
    imageIds = []
    data = []
    for imagePath in imagePaths:
        # Read Image Id
        imageId = int(imagePath.split(os.path.sep)[-1].split(".")[0])
        imageIds.append(imageId)
        
        # Extract data
        im = cv2.imread(imagePath)
        # Extract feature vectors from images using 'image_to_feature_vector'
        features = image_to_feature_vector(im)
        data.append(features)
    
    # data = [image_to_feature_vector(cv2.imread(imagePath)) for imagePath in imagePaths]
    # Normalize the data (by highest intesity - '255.0')
    data = np.array(data)/ 255.0    
    return imageIds, data


# In[9]:

data, labels = load_data()

''' Requires some data processing '''
le = LabelEncoder()
labels = le.fit_transform(labels)
num_classes = np.unique(labels).shape[0]
labels = np_utils.to_categorical(labels, num_classes)


# In[16]:

print '[INFO] Constructing Training and Validation Split'
train_x, valid_x, train_y, valid_y =  train_test_split(data, labels, test_size=0.3, random_state=21)


# In[13]:

class LeNetDroput:
    @staticmethod
    def build(width, height, depth, classes, 
              nb_filters=32, kernel_size=(3, 3), pool_size=(2,2), weightsPath=None):
        # Initialzing the model
        model = Sequential()
        
        ''' First set of CONV => RELU'''
        # Add Convolution Layers '20' filters and receptive filed of size (5 x 5)
        ''' Note - You need to provide the 'input_shape' only for the first layer in keras
           BORDER_MODE - 'valid' => No zero padding to the input,
                         'same'  => Padding such that input_dim=output_dim  '''
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], 
                                border_mode='valid', input_shape=(depth, height, width)))
        # Add Recitified Linear Units
        model.add(Activation('relu'))
        # Then Max Pooling with 'pool size (2 x 2)'
        model.add(MaxPooling2D(pool_size=pool_size))
        
        ''' Second set of CONV => RELU => pooling'''
        # Add Convolution Layers '20' filters and receptive filed of size (5 x 5)
        model.add(Convolution2D(50, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Then Max Pooling with 'pool size (2 x 2)'
        model.add(MaxPooling2D(pool_size=pool_size))
        # We add a drop out layer here - with 0.25 dropout
        model.add(Dropout(0.25))
                
        ''' Fully Connected Layers, followed by 'RELU' layer '''
        # First flatten the input 
        model.add(Flatten())
        # Add FC (Dense) Layer with 'output_dim' - 500
        model.add(Dense(128))
        # Add Recitified Linear Units
        model.add(Activation('relu'))
        # Again add a drop out layer here - with 0.5 dropout
        model.add(Dropout(0.5))
        
        ''' Final Soft Max Layer '''
        # FC Layer with 'output_dim' - number_of_classes
        model.add(Dense(classes))
        # Add Final Soft Max Activation
        model.add(Activation('softmax'))
        
        # Load weights if given
        if weightsPath is not None:
            model.load_weights(weightsPath)
            
        # Return the constructed model
        return model


# In[14]:

# Function to train the model
def train_model(train_x, train_y, valid_x, valid_y,
                weightsPath='../weights/lenet_weights_cifar.hdf5', opt='sgd', epochs=20,
                mini_batch_size=128, loss="categorical_crossentropy"):
    # Create Model
    model = LeNetDroput.build(width=32, height=32, depth=3, classes=10, 
                         weightsPath=weightsPath)
    # Configure the model for training
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.fit(train_x, train_y, batch_size=mini_batch_size, nb_epoch=epochs, verbose=1)
    
    # show the accuracy on validation data
    print '[INFO] Evaluating Validation data..'
    loss, accuracy = model.evaluate(valid_x, valid_y, batch_size=mini_batch_size,
                                    verbose=1)
    print '[INFO] Accuracy:{:.2f}%'.format(accuracy*100)
    return model

# # In[18]:

batch_size = 32
model = train_model(train_x, train_y, valid_x, valid_y, mini_batch_size=batch_size, 
                    weightsPath=None, opt='adadelta')


# In[19]:

model_default = train_model(train_x, train_y, valid_x, valid_y,
                    weightsPath=None, opt='adadelta')


# In[20]:

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
batch_size = 32
model_sgd = train_model(train_x, train_y, valid_x, valid_y, mini_batch_size=batch_size, 
                    weightsPath=None, opt=sgd)


# In[21]:

model_sgd_default = train_model(train_x, train_y, valid_x, valid_y,
                    weightsPath=None, opt=sgd)


In[31]:

imageIds, test_data = load_test_data('../data/test/')

batch_size = 32
test_predictions_1 = model.predict_classes(test_data, batch_size=batch_size, verbose=1)


# In[24]:

write_output('../output/leNet_droput_adadelta_bs_32.csv',imageIds, test_predictions_1)


# In[32]:

batch_size = 128
test_predictions_2 = model_default.predict_classes(test_data, batch_size=batch_size, verbose=1)


# In[33]:

write_output('../output/leNet_droput_adadelta.csv',imageIds, test_predictions_2)


# In[34]:

batch_size = 32
test_predictions_3 = model_sgd.predict_classes(test_data, batch_size=batch_size, verbose=1)


# In[35]:

write_output('../output/leNet_droput_sgd_bs_32.csv',imageIds, test_predictions_3)


# In[36]:

batch_size = 128
test_predictions_4 = model_sgd_default.predict_classes(test_data, batch_size=batch_size, verbose=1)


# In[37]:

write_output('../output/leNet_droput_sgd.csv',imageIds, test_predictions_4)
