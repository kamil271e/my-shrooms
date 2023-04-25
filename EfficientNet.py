from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.applications import EfficientNetB7
from keras.applications.efficientnet import preprocess_input
from NetUtils import NetUtils

class EfficientNet(NetUtils):
    def __init__(self, input_shape, num_classes, use_callbacks=False, load_weights = None):
        self.input_shape = input_shape
        self.efficient_base = EfficientNetB7(weights='imagenet',
                                    pooling='avg',
                                    include_top=False,
                                    input_shape=input_shape)
        self.efficient_base.trainable = False
        self.model = Sequential()
        self.model.add(self.efficient_base)
        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(num_classes, activation='softmax'))
        
        self.model.compile(loss='categorical_crossentropy',
                          metrics=['accuracy'],
                          optimizer='adam')
        
        if load_weights:
            self.model.load_weights(load_weights)

    def create_generators(self, train_dir, val_dir, test_dir, batch_size, data_augm=False, preprocessing_function=None):
        super().create_generators(train_dir, val_dir, test_dir, batch_size, data_augm, 
                                  preprocessing_function=preprocess_input)
    