# This is a script that shows how to use the code. It is not complete and will
# most likely not run out-of-the-box, but it will give you an idea.

import argparse
from lib import dataset
from lib import training
from lib import utils
from networks import basic
from networks import advanced
import os
from tensorflow import keras


parser = argparse.ArgumentParser(description='HAM10000 lesion classification')
parser.add_argument('--project_folder', default='./', help='project folder')
args = parser.parse_args()

# set up problem - download and unzip data, etc.
folder_project = args.project_folder
folder_data, folder_image, folder_models, metadata = utils.setup_problem(
        folder_project, download=True)

# first block of experiments - only image data ================================
# split dataset into disjoint train, validation and test
train_gen, test_gen, validate_gen = utils.setup_data_generators(
        metadata,
        folder_image,
        field_class="dx",
        batch_size=50,
        aux_data=False
        )

# first experiment - vanilla DCNN =============================================
model = basic.conv_net(num_filters=64, full_model=True, num_pool=4,
                      fully_connected_head=True, output_classes=7,
                      batchnorm=True)

model, history = training.compile_and_train(
        model, 
        train_gen, 
        validate_gen, 
        os.path.join(folder_models, "vanilla-cnn.h5"),
        multithreading = True
    )

pred, true = utils.collect_predictions(model, test_gen)
utils.plot_confusion_matrix(pred, true, classes=test_gen.unique_classes)
print("The first 3 top-k accuracies: ", utils.top_k_accuracies(pred, true))

# same for the remaining architectures ...
model = advanced.dense_net(num_block=6,
                           block_growth_rate=64,
                           num_pool=5,
                           pool_bottleneck_compression=0.75,
                           fully_connected_head=True, 
                           output_classes=7,
                           full_model=True)

# etc... let's jump

# final experiment - transfer learning on VGG with auxiliary data =============
# normalize age
metadata.age = metadata.age - metadata.age.min()
metadata.age = metadata.age / metadata.age.max()

# get generators
train_gen, test_gen, validate_gen = utils.setup_data_generators(
        metadata,
        folder_image,
        field_class="dx",
        batch_size=50,
        aux_data=True
        )

# get base model
base_model = keras.applications.VGG16(include_top=False, classes=7)
input_sex = keras.layers.Input(shape=(2,))
input_age = keras.layers.Input(shape=(1,))
encoded_sex = keras.layers.Dense(64, activation='relu')(input_sex)
output = basic.global_max_head(num_classes=7,
                               num_steps=2,
                               num_units=1024,
                               aux_inputs=(input_sex, input_age),
                               batchnorm=True)(base_model.output)
model = keras.models.Model(inputs=[base_model.input,input_sex,input_age], 
                              outputs=[output])

# freeze VGG
for layer in base_model.layers:
    layer.trainable = False

model, history = training.compile_and_train(
        model, 
        train_gen, 
        validate_gen, 
        os.path.join(folder_models, "vgg-transfer-aux.h5"),
        multithreading = True
)

# evaluate
pred, true = utils.collect_predictions(model, test_gen)
utils.plot_confusion_matrix(pred, true, classes=test_gen.unique_classes)
print("The first 3 top-k accuracies: ", utils.top_k_accuracies(pred, true))
