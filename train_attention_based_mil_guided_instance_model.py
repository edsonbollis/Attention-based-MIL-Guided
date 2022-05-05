from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import sys

import gc

import os
import os.path
from datetime import time

sys.path.append(".")
import tensorflow
import tensorflow.keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from image_data_generator import ImageDataGenerator
from tensorflow.keras import regularizers
import pandas as pd
from tensorflow.keras import backend as K
from EfficientNet_tf import _preprocess_input
from two_wam import Two_WAM
import multiprocessing

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

TL_WEIGHTS = 'pests.h5'


width_, height_ = None, None
size = None, None
layers = 3
classes = 2
epochs = None
batch_size = None
batch_balanced = None

parser = argparse.ArgumentParser(description='Attention-based MIL-Guided Instance Model Training')
parser.add_argument('folder', metavar='Folder',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('folder_instances', metavar='Folder',
                    help='path to instances (e.g. ../data/')
parser.add_argument('--image-size', '-s', default=400, type=int,
                    metavar='N', help='image size (default: 4000)')
parser.add_argument('--epochs', '-e', default=200, type=int,
                    metavar='N', help='training epochs (default: 200)')
parser.add_argument('--batch-size', '-b', default=16, type=int,
                    metavar='N', help='batch size (default: 8)')
parser.add_argument('--no-batch-balanced', '-nb', action='store_true', default=False,
                help='batch balanced (default: False)')
parser.add_argument('--noise', '-no', action='store_true', default=False,
                help='use noise data to train (default: False)')

args = parser.parse_args()

x_load, x_val_load = None, None

labels= ['Negative', 'Mite']

folder = args.folder
folder_ = args.folder_instances

y_csv = folder + 'pests_train_original.csv'
y_validation_csv = folder + 'pests_validation_original.csv'

y_csv_no_blurred = folder + 'pests_train_no_blured_manual.csv'
y_validation_no_blurred_csv = folder + 'pests_validation_no_blured_manual.csv'

def load_data(directory):
    load = pd.read_csv(directory,delim_whitespace=False,header=1,delimiter=',',
                       names = ['images', labels[1],labels[0]])

    class_vet = [labels[i] for i in load['Mite']]

    load['classes'] = np.array(class_vet)
    load['classes_id'] = load['Mite'].to_numpy()

    count = 0
    print(load)
    for i in load['images']:
        if os.path.isfile(folder + i):
            count += 1
    print("Images found:", count, " Images remaining:", len(load['images']) - count)
    return load

def change_crop(data):

    df = pd.DataFrame()
    class_vet = []
    class_number_vet = []
    images = []
    count_0 = 0
    count_1 = 0
    for i, line in data.iterrows():
        if line['classes_id'] == 0:
            idx = 5
            count_0+=idx
        else:
            idx = 2
            count_1 += idx
        for j in range(idx):
            class_vet.append(line['classes'])
            class_number_vet.append(line['classes_id'])
            images.append('cutted_'+ str(j) + "_" + line['images'])
    print("\n\n######### Class Negative:", count_0, "Class Positive:", count_1)
    print("\n\n")
    df['classes'] = np.array(class_vet)
    df['classes_id'] = np.array(class_number_vet)
    df['images'] = np.array(images)

    count = 0
    for i in df['images']:
        if os.path.isfile(folder + i):
            count += 1
    print("Images found:", count, " Images remaining:", len(df['images']) - count)
    return df

def f1():
    recall = tensorflow.keras.metrics.Recall()
    precision = tensorflow.keras.metrics.Precision()

    def f1(y_true, y_pred):
        precision_ = precision(y_true,y_pred)
        recall_ = recall(y_true,y_pred)
        return 2 * (precision_ * recall_) / (precision_ + recall_ + K.epsilon())

    return f1


def load_model_first(weights='imagenet'):

    print("shape=",(width_, height_, layers))
    input_tensor = Input(shape=(width_, height_, layers))  # this assumes K.image_data_format() == 'channels_last'

    from EfficientNet_tf import EfficientNetB32
    initial_model = EfficientNetB32(input_tensor=input_tensor, default_resolution=width_, weights=weights,
                                    include_top=False, input_shape=(width_, height_, 3),
                                    spatial_dropout=True)

    attention, _ = Two_WAM()(initial_model.output)

    op = GlobalAveragePooling2D()(attention)

    op = tensorflow.keras.layers.Dropout(0.3, name='dropout_final')(op)
    output_tensor = Dense(classes, activation='softmax', use_bias=True, name='Logits')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    for layer in model.layers:
        layer.trainable = True
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = regularizers.l1_l2(0.3)


    model.compile(optimizer=tensorflow.keras.optimizers.Adadelta(lr=0.1, rho=0.9, epsilon=0.9, decay=0.0005),
                  loss='categorical_crossentropy',
                  metrics=[categorical_accuracy, f1()])


    model.summary()


    return model

def preprocessing_function(x_in):
    x_ = _preprocess_input(x_in)
    # x = ((x/255) - mean) / std
    # x = (x / 255)
    return x_

def scheduler(epoch, lr):
    new_lr = lr
    if epoch != 0 and epoch != 1 and epoch%50 == 0:
        new_lr = lr / 10
    print('Learn rate:',new_lr)
    return new_lr


def train_model( x, x_val, class_weights = None, train_path = None, train_index = None,  imagenet = None):
    # checkpoint



    if train_index == None:
        if os.path.isfile(TL_WEIGHTS):
            print("################## \n\nLoading " + TL_WEIGHTS + "\n\n")

            # Put here the name of the weights that you would like to use

            print("##################")
            model = load_model_first()
            model.load_weights(TL_WEIGHTS)
        else:
            if imagenet == None or not imagenet:
                    model = load_model_first()
            else:
                model = load_model_first('imagenet')
        tensorboard_log_dir = "./{}".format(time())
        log_dir = ''
    else:

        try:
            if not os.path.exists(train_path):
                os.mkdir(train_path)
            if not os.path.exists(train_path + "/run_" + train_index):
                os.mkdir(train_path + "/run_" + train_index)
        except OSError:
            print("Creation of the directory %s failed" % train_path)
        else:
            print("Successfully created the directory %s " % train_path)

        if os.path.isfile(train_path + "/run_" + train_index + '/' + TL_WEIGHTS):
            print("################## \n\nLoading " + TL_WEIGHTS + "\n\n")
            print("##################")
            model = load_model_first()
            model.load_weights(train_path + "/run_" + train_index + '/' + TL_WEIGHTS)
        else:
            if imagenet == True:
                model = load_model_first('imagenet')
            else:
                model = load_model_first()

        tensorboard_log_dir = train_path + "/run_" + train_index + "/{}".format(time())
        log_dir = train_path + "/run_" + train_index + "/"


    if(train_index != None):
        filepath = log_dir + '/weights-improvement-{epoch:02d}-{val_f1:.2f}.hdf5'
    else :
        filepath = "weights-improvement-2-{epoch:02d}-{val_f1:.2f}.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir=tensorboard_log_dir)

    es = EarlyStopping(monitor='val_categorical_accuracy', patience=25, mode='auto', min_delta=0.001)

    # change_lr = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint, tensorboard, es]

    reindex = []
    for i in range(len(x)//batch_size):
        index=np.array([i for i in range(batch_size)])
        np.random.shuffle(index)
        index = index + batch_size*i
        reindex.append(index)
    index = np.array(reindex)
    index = index.flatten()
    x = x.reindex(index)


    reindex = []
    for i in range(len(x_val) // batch_size):
        index = np.array([i for i in range(batch_size)])
        np.random.shuffle(index)
        index = index + batch_size * i
        reindex.append(index)
    index = np.array(reindex)
    index = index.flatten()
    x_val = x_val.reindex(index)

    # unic = []
    # for i in x['images']:
    #     if i not in unic:
    #         unic.append(i)
    #     else:
    #         print(i)

    print(x, x_val)


    datagen = ImageDataGenerator(
        zoom_range=0.4,
        rotation_range=15,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=(-4, 4),
        height_shift_range=(-4, 4),
        preprocessing_function=preprocessing_function
    )


    test_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=(-4, 4),
        height_shift_range=(-4, 4),
        preprocessing_function=preprocessing_function)

    print(x, x_val)
    print("Shape=", width_, height_)
    train_datagen = datagen.flow_from_dataframe(x, directory=folder_, x_col='images', y_col='classes',
                                                class_mode='categorical', target_size=(width_, height_),
                                                batch_size=batch_size, class_weights= dict,
                                                classes=labels,
                                                shuffle=True)

    validation_datagen = test_datagen.flow_from_dataframe(x_val,directory=folder_, x_col='images', y_col='classes',
                                                          class_mode='categorical',target_size=(width_,height_),
                                                          batch_size=batch_size,
                                                          classes=labels,
                                                          shuffle=True)
    model.fit_generator(train_datagen,
                        steps_per_epoch=len(x) / (batch_size),
                        epochs=epochs,
                        workers=8,
                        use_multiprocessing=False,
                        verbose=1,
                        validation_data= validation_datagen,
                        callbacks=callbacks_list, class_weight = class_weights,
                        validation_steps=20)

    model.save(log_dir + 'pests.h5_2')



def dfprocessing_func(data):
    batch, j, x, x_val = data[0], data[1], x_load, x_val_load
    train_path = ''

    train_path = train_path + "attention_based_mil_guided_instance_model"

    train_index = str(j)

    print("\n\n\n*************Experiment ", train_path, " ", train_index, "***********\n")


    x = x.sample(frac=1).reset_index(drop=True)
    y = np.array(x['classes_id'])

    vet_indexes = [[] for i in range(classes)]

    # normalize the batch
    if batch:
        y_aux = y.flatten()
        for i in range(y_aux.shape[0]):
            vet_indexes[y_aux[i]].append(i)

        bath_indexes = []
        vet_aux = [len(vet) for vet in vet_indexes]
        vet_aux.sort()
        # for i in range(vet_aux[len(vet_aux)*2//3]):
        for i in range(max(vet_aux)):
            for j in range(classes):
                bath_indexes.append(vet_indexes[j][i%len(vet_indexes[j])]);
        vet = []
        for rows in x.itertuples():
            vet.append([rows.images,  rows.classes, rows.classes_id])
        vet = np.array(vet, dtype=object)[bath_indexes]
        x = pd.DataFrame({'images': vet[:, 0],
                          'classes': vet[:, 1],
                          'classes_id': vet[:,2]})

    y_aux = y.flatten()
    for i in range(y_aux.shape[0]):
        vet_indexes[y_aux[i]].append(i)

    vet_aux = [len(vet) for vet in vet_indexes]


    for i, val in enumerate(vet_indexes):
        dict[labels[i]]= max(vet_aux) - len(val) + 1


    # put multiclass form
    # mat = np.array([ i for i in x['classes']])
    # for idx, label_name in enumerate(labels):
    #     x[label_name] = mat[:, idx]



    x_val = x_val.sample(frac=1).reset_index(drop=True)
    y = np.array(x_val['classes_id'])

    vet_indexes = [[] for i in range(classes)]
    # normalize the batch
    if batch:
        y_aux = y.flatten()
        for i in range(y_aux.shape[0]):
            vet_indexes[y_aux[i]].append(i)

        bath_indexes = []
        vet_aux = [len(vet) for vet in vet_indexes]
        for i in range(max(vet_aux)):
            for j in range(classes):
                bath_indexes.append(vet_indexes[j][i % len(vet_indexes[j])]);
        vet = []
        for rows in x_val.itertuples():
            vet.append([rows.images, rows.classes, rows.classes_id])
        vet = np.array(vet, dtype=object)[bath_indexes]

        x_val = pd.DataFrame({'images': vet[:, 0],
                              'classes': vet[:, 1],
                              'classes_id': vet[:,2]})

    # mat = np.array([i for i in x['classes']])
    # for idx, label_name in enumerate(labels):
    #     x_val[label_name] = mat[:, idx]


    print("x", x.shape)
    print("x_val", x_val.shape)

    train_model(x, x_val, train_index=train_index, train_path=train_path, imagenet=False)
    del x, x_val
    print('\n\n\n')
    gc.collect()




if __name__ == '__main__':


    dict = {}

    width_, height_ = args.image_size, args.image_size
    size = width_, height_

    epochs = args.epochs
    batch_size = args.batch_size
    if not args.no_batch_balanced:
        batch_balanced = True
    else:
        batch_balanced = False

    if not args.noise:
        y_csv = folder + 'pests_train_no_blured_manual.csv'
        y_validation_csv = folder + 'pests_validation_no_blured_manual.csv'
    else:
        y_csv = folder + 'pests_train_original.csv'
        y_validation_csv = folder + 'pests_validation_original.csv'

    for fold in range(1, 2):
        x_ = load_data(y_csv).sample(frac=1, random_state=fold)
        x_val = change_crop(x_[80 * x_.shape[0] // 100:])
        x_ = change_crop(x_[:80 * x_.shape[0] // 100])

        x_load, x_val_load = x_, x_val

        print("Image number", x_.shape, x_val.shape)

        # multprocess to release data
        p = multiprocessing.Pool(1)
        p.map(dfprocessing_func, [[batch_balanced, fold]])
        # dfprocessing_func([batch, expoent_index, fold])
        p.terminate()
        p.join()

