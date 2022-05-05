import argparse
import shutil
import sys
import os

import numpy as np
import cv2
import tensorflow
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.append(".")
from EfficientNet_tf import _preprocess_input
import random
from shutil import copy
from two_wam import Two_WAM

classes = 2

parser = argparse.ArgumentParser(description='Attention-based MIL-Guided Patch-SaliMap cuts')
parser.add_argument('folder', metavar='Folder',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('new_folder', metavar='New Folder',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('weights', metavar='Weights',
                    help='Trained Weights')
parser.add_argument('--image-size', '-s', default=1200, type=int,
                    metavar='N', help='image size (default: 1200)')
parser.add_argument('--patch-size', '-p', default=400, type=int,
                    metavar='N', help='patch size (default: 400)')
parser.add_argument('--instance-number', '-i', default=5, type=int,
                    metavar='N', help='patch number (default: 5)')
parser.add_argument('--batch-size', '-b', default=8, type=int,
                    metavar='N', help='batch size (default: 8)')


args = parser.parse_args()

labels= ['Negative', 'Mite']

folder = args.folder

y_csv = folder + 'pests_train_original.csv'
y_validation_csv = folder + 'pests_validation_original.csv'
y_test_csv = folder + 'pests_test_original.csv'

new_folder = args.new_folder

H, W = args.image_size, args.image_size  # Input shape, defined by the model (model.input_shape)
# Define model here ---------------------------------------------------


def load_data(directory):
    load = pd.read_csv(directory,delim_whitespace=False,header=1,delimiter=',',
                       names = ['images', labels[1],labels[0]])

    class_vet = [labels[i] for i in load['Mite']]

    load['classes'] = np.array(class_vet)
    load['classes_id'] = load['Mite'].to_numpy()

    count = 0
    for i in load['images']:
        if os.path.isfile(folder + i):
            count += 1
    print("Images found:", count, " Images remaining:", len(load['images']) - count)
    return load


def load_model_first():

    print("shape=",(W, H, 3))
    input_tensor = Input(shape=(W, H, 3))  # this assumes K.image_data_format() == 'channels_last'

    from EfficientNet_tf import EfficientNetB32
    initial_model = EfficientNetB32(input_tensor=input_tensor, default_resolution=W, weights=None,
                                    include_top=False, input_shape=(W, H, 3),
                                    spatial_dropout=True, dropout=0.4)
    attention, maps = Two_WAM()(initial_model.output)

    op = GlobalAveragePooling2D()(attention)

    output_tensor = Dense(classes, activation='softmax', use_bias=True, name='Logits')(op)

    model = Model(inputs=input_tensor, outputs=[output_tensor, maps])

    # model.summary()

    model.load_weights(weights)

    return model



# ---------------------------------------------------------------------

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(W, H))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = _preprocess_input(x)
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def twowam_batch(input_model, images):
    """Two-WAM method for visualizing input saliency."""

    _, activation = input_model.predict(images)

    # Process CAMs
    new_cams = np.empty((images.shape[0], W, H))
    for i in range(new_cams.shape[0]):
        cam_i = activation[i] - activation[i].mean()
        cam_i = np.reshape(cam_i,(cam_i.shape[0],cam_i.shape[1]))
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (W, H), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    return new_cams

# This Patch-SaliMap was modified to work with batch.
# The base of the algorithm is the same
# Here it creates the the Grad-CAM Image and the Patches
def Patch_SaliMap(model, img_path, cls=-1):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class).
    """
    patch_size = args.patch_size
    patch_size = patch_size // 2

    initial = 0
    batch_size = args.batch_size
    cuts=5
    img_name = []
    img_mite_location_x = []
    img_mite_location_y = []


    for n in range((len(img_path) - initial) // batch_size):
        preprocessed_input = []
        images = []
        for i, img_ in enumerate(img_path[initial:initial+batch_size]):
            preprocessed_input.append(load_image(folder + img_))
            images.append(img_)
            for j in range(cuts):
                img_name.append(img_)
        preprocessed_input = np.concatenate(preprocessed_input,axis=0)

        gradcam_batch = twowam_batch(model, preprocessed_input)

        print("Grad-cam batch len: ", gradcam_batch.shape[0])
        for i in range(gradcam_batch.shape[0]):
            img_ = images[i]
            gradcam = gradcam_batch[i,:,:]
            cls_actual = cls[initial + i]

            jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
            print("#### jetcam.shape",jetcam.shape)
            ind = np.unravel_index(np.argmax(gradcam, axis=None), gradcam.shape)
            jetcam[ind[0]-10:ind[0]+10,ind[1]-10:ind[1]+10] = 0
            jetcam = (np.float32(jetcam) + load_image(folder + img_, preprocess=False)) / 2
            cv2.imwrite(new_folder + 'gradcam_' + img_, np.uint8(jetcam))

            for j in range(cuts):
                ind = np.unravel_index(np.argmax(gradcam, axis=None), gradcam.shape)
                image_ = image.img_to_array(load_image(folder + img_, preprocess=False))

                if cls_actual == 1:
                    ind = (int((ind[0] / gradcam.shape[0]) * image_.shape[0]),
                           int((ind[1] / gradcam.shape[1]) * image_.shape[1]))
                else:
                    ind = (random.randint(patch_size,W-patch_size),random.randint(patch_size,H-patch_size))

                img_mite_location_x.append(ind[0])
                img_mite_location_y.append(ind[1])

                vet_ = [ind[0], ind[1]]
                if ind[0] - patch_size < 0:
                    vet_[0] -= ind[0] - patch_size
                if ind[1] - patch_size < 0:
                    vet_[1] -= ind[1] - patch_size
                if ind[0] + patch_size >= image_.shape[0]:
                    vet_[0] += image_.shape[0] - ind[0] - patch_size + 1
                if ind[1] + patch_size >= image_.shape[0]:
                    vet_[1] += image_.shape[1] - ind[1] - patch_size + 1

                image_ = image_[vet_[0] - patch_size: vet_[0] + patch_size, vet_[1] - patch_size: vet_[1] + patch_size, :]
                cv2.imwrite(new_folder + 'cutted_' + str(j) + "_" + img_, np.uint8(image_))
                gradcam[vet_[0] - patch_size: vet_[0] + patch_size, vet_[1] - patch_size: vet_[1] + patch_size] = -1e7

        initial += batch_size
        print("Image" + str(initial) + " from " + str(len(img_path)))

    preprocessed_input = []
    images = []
    for i, img_ in enumerate(img_path[initial:len(img_path)]):
        preprocessed_input.append(load_image(folder + img_))
        images.append(img_)
        for j in range(cuts):
            img_name.append(img_)
    preprocessed_input = np.concatenate(preprocessed_input, axis=0)

    gradcam_batch = twowam_batch(model, preprocessed_input)

    print("Grad-cam batch len: ", gradcam_batch.shape[0])
    for i in range(gradcam_batch.shape[0]):
        img_ = images[i]
        gradcam = gradcam_batch[i,:,:]
        cls_actual = cls[initial + i]

        for j in range(cuts):
            ind = np.unravel_index(np.argmax(gradcam, axis=None), gradcam.shape)
            image_ = image.img_to_array(load_image(folder + img_, preprocess=False))

            if cls_actual == 1:
                ind = (int((ind[0] / gradcam.shape[0]) * image_.shape[0]),
                       int((ind[1] / gradcam.shape[1]) * image_.shape[1]))
            else:
                ind = (random.randint(patch_size, W - patch_size), random.randint(patch_size, H - patch_size))

            img_mite_location_x.append(ind[0])
            img_mite_location_y.append(ind[1])

            vet_ = [ind[0], ind[1]]
            if ind[0] - patch_size < 0:
                vet_[0] -= ind[0] - patch_size
            if ind[1] - patch_size < 0:
                vet_[1] -= ind[1] - patch_size
            if ind[0] + patch_size >= image_.shape[0]:
                vet_[0] += image_.shape[0] - ind[0] - patch_size + 1
            if ind[1] + patch_size >= image_.shape[0]:
                vet_[1] += image_.shape[1] - ind[1] - patch_size + 1

            image_ = image_[vet_[0] - patch_size: vet_[0] + patch_size, vet_[1] - patch_size: vet_[1] + patch_size, :]
            cv2.imwrite(new_folder + 'cutted_' + str(j) + "_" + img_, np.uint8(image_))
            gradcam[vet_[0] - patch_size: vet_[0] + patch_size, vet_[1] - patch_size: vet_[1] + patch_size] = -1e7

    initial = len(img_path)
    print("Image" + str(initial) + " from " + str(len(img_path)))

    df = pd.DataFrame()
    print("Len name array",len(img_name))
    df['images'] = np.array(img_name)
    print("Len x locations", len(img_mite_location_x))
    df['mite_location_x'] = np.array(img_mite_location_x)
    print("Len y locations", len(img_mite_location_y))
    df['mite_location_y'] = np.array(img_mite_location_y)
    df.to_csv(path_or_buf=new_folder +'pests_database_location_validation.csv')



if __name__ == '__main__':

    weights = args.weights

    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

    model = load_model_first()

    for file in [y_csv, y_validation_csv, y_test_csv]:
        x_ = load_data(file).sample(frac=1, random_state=5)

        copy(file, new_folder)

        print("Input preprocess", x_.shape)

        Patch_SaliMap(model, img_path=x_['images'].to_list(), cls=x_['classes_id'].to_list())
