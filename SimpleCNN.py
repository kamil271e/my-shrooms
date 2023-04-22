from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, CategoricalAccuracy, Precision, Recall, F1Score, AUC

class SimpleCNN:
    def __init__(self, input_shape, num_classes, load_weights = None):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Conv2D(64, (3,3), activation='relu'))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Conv2D(64, (3,3), activation='relu'))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Conv2D(128, (3,3), activation='relu'))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        
        self.metrics = [
            #TruePositives(name='tp'),
            #FalsePositives(name='fp'),
            #TrueNegatives(name='tn'),
            #FalseNegatives(name='fn'),
            CategoricalAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            F1Score(name='f1'),
            AUC(name='auc'),
            AUC(name='prc', curve='PR')
        ]               
                       
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=1e-4),
                           metrics=self.metrics)
        if load_weights:
            self.model.load_weights(load_weights)
                       

    #def _preprocess_data(self, x_train, y_train, x_test, y_test):
        # Reshape input data and normalize pixel values
    #    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    #    x_test = x_test.reshape(-1, 784).astype('float32') / 255

        # Standardize input data using the scaler
    #    self.scaler.fit(x_train)
    #    x_train = self.scaler.transform(x_train)
    #    x_test = self.scaler.transform(x_test)

        # Convert labels to one-hot encoded vectors
    #    y_train = to_categorical(y_train, num_classes)
    #    y_test = to_categorical(y_test, num_classes)

    #    return x_train, y_train, x_test, y_test

    def load_generators(self, train_generator, val_generator):
        self.train_generator = train_generator
        self.val_generator = val_generator
                       
    def load_callbacks(self, callback_list):
        self.callback_list = callback_list
                       
    def train(self, steps_per_epoch, epochs, validation_steps):
        return self.model.fit(
           self.train_generator,
           steps_per_epoch = steps_per_epoch,
           epochs = epochs,
           callbacks = self.callback_list,
           validation_data = self.val_generator,
           validation_steps = validation_steps
        )
