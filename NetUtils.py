import numpy as np
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 
# Each model inherits functionality from this class
#

class NetUtils:
    def __init__(self):
        self.model = Sequential()
    
    def create_generators(self, train_dir, val_dir, test_dir, batch_size, data_augm=False, preprocessing_function=None):
        if data_augm:
            train_datagen = ImageDataGenerator(
                preprocessing_function=preprocessing_function,
                rotation_range=40,
                height_shift_range=0.2,
                width_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

        test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical'
        )

        self.val_generator = test_datagen.flow_from_directory(
            val_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical'
        )
                
        
    def create_callbacks(self, filepath):
        self.callback_list = [
            EarlyStopping(monitor='accuracy', patience=2.),
            ModelCheckpoint(filepath=filepath,
                       monitor='val_loss',
                       save_best_only=True),
            # ReduceLROnPlateau(monitor='val_loss', patience=2)
        ]
    
    
    def load_generators(self, train_generator, val_generator, test_generator):
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
    
    
    def load_callbacks(self, callback_list):
        self.callback_list = callback_list
        
    
    def train(self, steps_per_epoch, epochs, validation_steps):
        callback_list = None
        try:
            callback_list = self.callback_list
        except:
            print("There is no callbacks loaded")
            
        return self.model.fit(
           self.train_generator,
           steps_per_epoch = steps_per_epoch,
           epochs = epochs,
           callbacks = callback_list,
           validation_data = self.val_generator,
           validation_steps = validation_steps
        )
    
    
    def summary(self):
        return self.model.summary()
    
    
    def clf_report(self):
        y_pred = self.model.predict(self.test_generator)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = self.test_generator.classes
        class_names = list(self.test_generator.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=class_names)
        return report
    
    
    # preprocess input
    def extract_features(self, model_base, directory, target_size, sample_count, num_classes=9, batch_size=32):
        datagen = ImageDataGenerator(rescale=1./255)
        conv_output_shape = model_base.layers[-1].output_shape[1:]
        shape = tuple([sample_count] + list(conv_output_shape))
        features = np.zeros(shape=shape)
        labels = np.zeros(shape=(sample_count, num_classes))
        generator = datagen.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        i = 0
        # Remember that generator returns data infinitely - we need break statement
        for inputs_batch, labels_batch in generator:
            feature_batch = model_base.predict(inputs_batch, verbose=0)
            features[i * batch_size : (i+1) * batch_size] = feature_batch
            labels[i * batch_size : (i+1) * batch_size] = labels_batch
            i += 1
            if i * batch_size >= sample_count:
                break
        return features, labels