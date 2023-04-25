from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, CategoricalAccuracy, Precision, Recall, AUC
from NetUtils import NetUtils

class SimpleCNN(NetUtils):
    def __init__(self, input_shape, num_classes, load_weights = None):
        self.input_shape = input_shape
        
        self.model = Sequential()
        self.model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3,3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3,3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3,3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(num_classes, activation='softmax'))
        
        self.metrics = [
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            AUC(name='prc', curve='PR')
        ]               
                       
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=1e-4),
                           metrics=self.metrics)
        if load_weights:
            self.model.load_weights(load_weights)
            
                                              