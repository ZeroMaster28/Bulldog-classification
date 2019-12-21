from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

# ====== Specifying Model ======

num_classes = 2
resnet_weights_path = '../resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

# ====== Compile Model ======

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# ====== Fit Model ======

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        '../train',
        target_size=(image_size, image_size),
        batch_size=1,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '../validation',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=3,
        validation_data=validation_generator,
        validation_steps=1)
