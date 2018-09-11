# i need to this package for initial cnn or ann
# ther are 2 way for initializing nural nets one is a sequence of layers or as graph
# since cnn is a sequence of layers we use sequential package
from keras.models import Sequential
#convolution 2d is used to make first layer of cnn ie the convoultution step which we add convolution layer
# we are workng with images we need 2d rather than 3d which we may can use in 3d images.
from keras.layers import Conv2D
# max pooling is used in second step which is pooling step
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
#image preprocessing
from keras.preprocessing.image import ImageDataGenerator

#part 1 initializing the cnn
classifire = Sequential()

#step 1 -- Convolution

''' 
	convolution2D arguments details
    first parameter 		  = no of feature detecters. no of feature map is same as no of feature dectores we use 
    secon and third parameter = row and coloum of feature detectors  
    optional parameter        =  so not used here which is border mode use of border in image. it has default value of same which means that it has same value as in image
    fourth parameter          = input shape which is format of image. example if use color image we need 3 dimensional array and each array has 2d array it has rows and columns
    fifthe parameter          = activation function here we use rectifir because we need to confirm negetive function and ofcourse selecting a pixel is non linear so we use rectifier

'''
classifire.add(Conv2D(32,(3,3), input_shape=(64,64,3), activation='relu'))
# classifire.add(Activation('relu'))

#step 2 --pooling

""" what is pooling?
       reducing the size of convoluted arrat by striding of order to over the obove
	   pool_size = size of array
		
"""
classifire.add(MaxPooling2D(pool_size = (2,2)))
#adding other convolution nn to improve accuracy but this time we are using maxpooled images not the input images
classifire.add(Conv2D(32,(3,3), input_shape=(64,64,3), activation='relu'))
classifire.add(MaxPooling2D(pool_size = (2,2)))


# step 3 - flattening

''' convert above to one big ondimensional vectores'''
classifire.add(Flatten())

''' we have to use this as input of input vector of artifical nural network  because ann is a good for dealing with non linear problem'''


# full conntection
classifire.add(Dense(units = 128, activation = 'relu'))
classifire.add(Dense(units = 1, activation = 'sigmoid'))

classifire.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

#image preprocessing
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifire.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=800)

# part 3 - making new prediction
# todo add single image using kereas image processing image function
# then image to array function to convert image to array 
# then exapand the image to batch using numpy expand dimesion function