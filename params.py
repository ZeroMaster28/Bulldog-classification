import os

# project name
project_name = 'bulldog'

# list of categories names 
categories = ['english', 'french']

# number of classes
num_classes = len(categories)

# available images per class
images_per_class = 200

# image size for normalization
image_size = 224

# complete number of epochs
number_of_epochs = 3

# validation:train proportion
proportion = 5

# size of a training batch and epoch steps 
train_batch = 10
epoch_steps = 16
assert train_batch * epoch_steps == int(images_per_class*(proportion-1)/proportion)

# validation batch and validation steps
validation_batch = 40
validation_steps = 1
assert validation_steps * validation_batch == int(images_per_class/proportion)

# training with augumentation
with_augmentation = False

# relative path
relative_path = os.path.abspath('.')

# path to weights dir
weights_path = relative_path + '\\weights'

# weights file name
weights_file = '\\resnet50_weights.h5'

# path to train dir
train_path = relative_path + '\\train'

#path to validation dir
validation_path = relative_path + '\\validation'  

# path to additional testing purpose dir 
test_path = relative_path + '\\test'

# path to created models
model_dir = relative_path + '\\model'

# path for saving model weights
new_weights_path = model_dir + '\\' + project_name + '_weights.h5'

#path for saving model architecture
model_arch_path = model_dir + '\\' + project_name + '_model.json'

# whether to save the model
save_model = True
