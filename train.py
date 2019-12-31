import os
import params as pm
import numpy as np
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


# model specification

num_classes = 2
resnet_weights_path = pm.weights_path + '\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
model.add(Dense(num_classes, activation='softmax'))

# saying not to train first layer (ResNet) model. It is already trained
model.layers[0].trainable = False

# model compilation
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# image preprocessing
data_generator = None 
if pm.with_augmentation:
        data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, 
                horizontal_flip=True,
                width_shift_range = 0.2,
                height_shift_range = 0.2)
else:
        data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

def read_and_prep_images(img_path, img_names, img_height=pm.image_size, img_width=pm.image_size):
    array_images = []
    for img_name in img_names: 
            single_image_path = img_path + '\\' + img_name
            image = load_img(single_image_path, target_size=(img_height, img_width))
            array_images.append(img_to_array(image))
    output = preprocess_input(np.array(array_images))
    return(output)

# fitting the model
train_generator = data_generator.flow_from_directory(
        pm.train_path,
        target_size=(pm.image_size, pm.image_size),
        batch_size=pm.train_batch,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        pm.validation_path,
        target_size=(pm.image_size, pm.image_size),
        batch_size=pm.validation_batch,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        epochs=pm.number_of_epochs,
        steps_per_epoch=pm.epoch_steps,
        validation_data=validation_generator,
        validation_steps=pm.validation_steps)

# saving the model
if pm.save_model:
        model.save_weights(pm.new_weights_path)
        with open(pm.model_arch_path, 'w') as f:
                f.write(model.to_json())

