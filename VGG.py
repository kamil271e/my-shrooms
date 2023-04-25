from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from NetUtils import NetUtils

class VGG(NetUtils):
    def __init__(self, input_shape, num_classes, load_weights = None):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.vgg_base = VGG16(weights='imagenet',
                              include_top=False,
                              input_shape=input_shape,
                             pooling='avg')
        
        self.model = Sequential()
        self.model.add(self.vgg_base)
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.layers[0].trainable = False
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        
        if load_weights:
            self.model.load_weights(load_weights)
    
    def create_generators(self, train_dir, val_dir, test_dir, batch_size, data_augm=False, preprocessing_function=None):
            super().create_generators(train_dir, val_dir, test_dir, batch_size, data_augm, 
                                      preprocessing_function=preprocess_input)