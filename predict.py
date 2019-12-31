import os
import numpy as np
import pandas as pd
import params as pm

from keras.models import model_from_json
from keras.applications import ResNet50
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

# ====== Specyfing paths =======



test_images = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg'] # names of images to test the model


# ====== Specifying Model ======

model = None
with open(pm.model_arch_path, 'r') as f:
    model = model_from_json(f.read())
model.load_weights(pm.new_weights_path)

# ====== Preprocessing images ======


def read_and_prep_images(img_path, img_names, img_height=pm.image_size, img_width=pm.image_size):
    array_images = []
    for img_name in img_names: 
            single_image_path = img_path + '\\' + img_name
            image = load_img(single_image_path, target_size=(img_height, img_width))
            array_images.append(img_to_array(image))
    output = preprocess_input(np.array(array_images))
    return(output)

# ====== Postprocessing images ======

def show_results(test_images, predictions):
    df = pd.DataFrame(columns=['Image','English','French'])
    for i in range(0, len(predictions)):
        df.loc[i] = [test_images[i], predictions[i][0], predictions[i][1]]
    print(df)


# ====== Predictions for unseen data ======

test_data = read_and_prep_images(pm.test_path, test_images)
predictions = model.predict(test_data)
show_results(test_images, predictions)

