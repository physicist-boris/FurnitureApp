import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#Generate data for model (data augmentation)
def train_dataset_creation(path_to_image_data, batch_length):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        brightness_range = [0.6, 1.6],
        horizontal_flip=True,
        validation_split=0)


    train_dataset_generator = datagen.flow_from_directory(
        path_to_image_data,
        target_size=(256, 256),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=batch_length,
        shuffle=True,
        seed=42,
        subset='training',
        keep_aspect_ratio=False
    )
    return train_dataset_generator


def val_dataset_creation(path_to_image_data):
    val_dataset =  tf.keras.utils.image_dataset_from_directory(directory=path_to_image_data, labels='inferred',
                                                            label_mode='categorical', batch_size=25, seed=123,
                                                            validation_split=0.5,
                                                            subset="validation", interpolation='nearest')
    return val_dataset

#Create neural network
def cnn_neural_network():
    cnn_model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3), use_bias= True),
                                 tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', use_bias = True),tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', use_bias = True),tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(128, (3,3), activation='relu', use_bias = True), tf.keras.layers.Dropout( 0.2), tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(128, (3,3), activation='relu', use_bias = True),tf.keras.layers.Dropout( 0.2),tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(256, activation= 'relu', use_bias = True), tf.keras.layers.Dropout( 0.2),
                                 tf.keras.layers.Dense(3, activation='softmax', use_bias = True)])
    cnn_model.compile(optimizer =tf.keras.optimizers.Adadelta(),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=[tf.keras.metrics.categorical_accuracy])
    return cnn_model


#Create classifier training object
class FurnitureClassifier:  
    def __init__(self):
        self.model = cnn_neural_network()

    def evaluate_model(self, training_set, validatation_set, epochs, path_to_save):
        STEP_SIZE_TRAIN = training_set.n // training_set.batch_size
        try:
            history = self.model.fit_generator(training_set, epochs=epochs, steps_per_epoch=STEP_SIZE_TRAIN,
                                validation_data= validatation_set, validation_steps = 6)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label = 'test')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='lower right')
            plt.show()
            plt.plot(history.history['categorical_accuracy'], label='train')
            plt.plot(history.history['val_categorical_accuracy'], label='test')
            plt.xlabel('Epoch')
            plt.ylabel('accuracy')
            plt.legend(loc='lower right')
            plt.show()
        except KeyboardInterrupt:
            print("Training stopped during process")
        self.model.save(path_to_save)