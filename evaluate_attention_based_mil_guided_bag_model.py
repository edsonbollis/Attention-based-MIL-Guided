from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import tensorflow
import tensorflow.keras
from tensorflow.keras import Input
import numpy as np
import sys
sys.path.append('../..')
sys.path.append('.')
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from image_data_generator import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os.path
import pandas as pd
from EfficientNet_tf import _preprocess_input
from two_wam import Two_WAM
from tensorflow.keras import backend as K



width_, height_ = None, None
size = None, None

log_file = open("log_bag_model.log", "a")

mean = np.array([0.36993951, 0.4671942,  0.34614164])
std = np.array([0.14322622, 0.1613108,  0.14663414])

layers = 3

classes = 2


parser = argparse.ArgumentParser(description='Attention-based MIL-Guided Bag Model Evaluation')
parser.add_argument('folder', metavar='Folder', help='path to dataset (e.g. ../data/')
parser.add_argument('run', metavar='run_[number]', type=str, help='run_[number]')
parser.add_argument('--image-size', '-s', default=1200, type=int,
                    metavar='N', help='image size (default: 1200)')
parser.add_argument('--batch-size', '-b', default=8, type=int,
                    metavar='N', help='batch size (default: 8)')

args = parser.parse_args()

folder = args.folder

y_csv = folder + 'pests_train_original.csv'
y_validation_csv = folder + 'pests_validation_original.csv'
y_test_csv = folder + 'pests_test_original.csv'

labels= ['Negative', 'Mite']


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

def f1():
    recall = tensorflow.keras.metrics.Recall()
    precision = tensorflow.keras.metrics.Precision()

    def f1(y_true, y_pred):
        precision_ = precision(y_true,y_pred)
        recall_ = recall(y_true,y_pred)
        return 2 * (precision_ * recall_) / (precision_ + recall_ + K.epsilon())

    return f1


def load_model_first(weights=None, expoent_index=None):
    print("shape=", (width_, height_, layers))
    input_tensor = Input(shape=(width_, height_, layers))  # this assumes K.image_data_format() == 'channels_last'

    from EfficientNet_tf import EfficientNetB32
    initial_model = EfficientNetB32(input_tensor=input_tensor, default_resolution=width_, weights=None,
                                    include_top=False, input_shape=(width_, height_, 3),
                                    spatial_dropout=True, dropout=0.4)

    attention, _ = Two_WAM()(initial_model.output)

    op = GlobalAveragePooling2D()(attention)

    output_tensor = Dense(classes, activation='softmax', use_bias=True, name='Logits')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = regularizers.l1_l2(0.3)

    model.compile(optimizer=tensorflow.keras.optimizers.Adadelta(lr=0.1, rho=0.9, epsilon=0.9, decay=0.0005),
                  loss='categorical_crossentropy',
                  metrics=[categorical_accuracy, f1()])

    # model.summary()

    return model


def train_model(x, x_val, x_test, model_name=None):
    # checkpoint

    model = load_model_first()

    model.load_weights(model_name)


    batch_size = args.batch_size
    print("\n*******Training")
    print("\n*******Training", file=log_file)
    accuracy_train = confusion_mat(model, batch_size, x, file='prediction_train')
    # accuracy_train = 0
    print("\n*******Validation")
    print("\n*******Validation", file=log_file)
    accuracy_validation = confusion_mat(model, batch_size, x_val, file='prediction_val')
    # accuracy_validation = 0
    print("\n*******Test")
    print("\n*******Test", file=log_file)
    accuracy_test = confusion_mat(model, batch_size, x_test, file='prediction_test')
    del model
    return accuracy_train, accuracy_validation, accuracy_test
    # return accuracy_test

def preprocessing_function(x_in):
    x_ = _preprocess_input(x_in)
    # x = ((x/255) - mean) / std
    # x = (x / 255)
    return x_

def confusion_mat(model, batch_size, x, file = None):

    print("###################Shape and len", len(x), x.shape, (width_, height_))

    imageDatagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

    print("shape=", (width_, height_, layers))



    _datagen = imageDatagen.flow_from_dataframe(x, directory=folder, x_col='images', y_col='classes',
                                                class_mode='categorical', target_size=(width_, height_),
                                                classes=labels,
                                                batch_size=batch_size, shuffle=False)
    # x_aux, y_aux = _datagen.next()
    # for i in range(x_aux.shape[0]):
    #     cls = np.array(x['classes'])
    #     print(i,cls,y_aux[i][cls[i]], np.argmax(y_aux[i]))
    #     plt.imshow(x_aux[i])
    #     plt.show()

    # y_val = np.roll(np.array(x['classes']), 0, axis=0)
    y_val = np.array(np.array(x['classes_id']))

    Y_val = y_val.flatten()

    # Confution Matrix and Classification Report
    y_pred = model.predict_generator(_datagen,steps = len(x)/batch_size, verbose=1, workers=8, use_multiprocessing=False)

    arg_max = np.argmax(y_pred, axis=1)

    val_max = np.max(y_pred, axis=1)

    mask = (val_max > 0.) * 1

    Y_pred = arg_max * mask

    if file != None:
        data_frame = pd.DataFrame()
        data_frame['images'] = np.array(x['images'][:y_pred.shape[0]])
        data_frame['classes_id'] = Y_val[:y_pred.shape[0]]
        data_frame['pred_negative'] = y_pred[:,0]
        data_frame['pred_positive'] = y_pred[:, 1]
        data_frame['pred_correct'] = Y_pred == Y_val[:y_pred.shape[0]]
        data_frame.to_csv(file + ".csv")

    print(x[:5], Y_val[:5], Y_pred[:5])

    vet = confusion_matrix(Y_val, Y_pred, labels=[i for i in range(0,classes)])

    val = sum([instance[i] for i,instance in enumerate(vet)])
    val /= len(x)

    accuracy = accuracy_score(Y_val, Y_pred, normalize=True, sample_weight=None)
    accuracy_no_norm = accuracy_score(Y_val, Y_pred, normalize=False, sample_weight=None)

    print('Confusion Matrix', np.array(Y_pred).shape, np.array(Y_val).shape)
    print('Confusion Matrix', np.array(Y_pred).shape, np.array(Y_val).shape, file=log_file)

    print("Accuracy", str(accuracy), 'No norm', str(accuracy_no_norm))
    print("Accuracy", str(accuracy), 'No norm', str(accuracy_no_norm), file=log_file)
    print('Classification Report')
    print('Classification Report', file=log_file)

    report = classification_report(Y_val, Y_pred, target_names=labels)
    print(report)
    print(report, file=log_file)
    log_file.flush()

    return str(accuracy).replace('.', ',')



def dfprocessing_func(data):

        model_name, x, x_val, x_test = data[0], data[1], data[2], data[3]

        print("\n********* Model Name:", model_name,
              "\t**********\n*********", datetime.datetime.now(), "\t**********\n")
        print("\n********* Model Name:", model_name,
              "\t**********\n*********", datetime.datetime.now(), "\t**********\n", file=log_file)
        weights_name.append(model_name)
        log_file.flush()

        train_val, validation_val, test_val = train_model(x, x_val, x_test, model_name=model_name)
        return [train_val, validation_val, test_val]

if __name__ == '__main__':


    model_names = []
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if "weights-improvement" in name:
                print(name)
                model_names.append(name)
    model_names.sort(key=lambda x: os.path.getmtime(x))

    weights_name = []
    validation_values = []
    train_values = []
    test_values = []

    fold = int(args.run[-1])

    x = load_data(y_csv).sample(frac=1, random_state=fold)
    x_val = x[80 * x.shape[0] // 100:]
    x_ = x[:80 * x.shape[0] // 100]
    x_test = load_data(y_validation_csv).sample(frac=1, random_state=5)

    print(x_, x_val, x_test)

    width_, height_ = args.image_size, args.image_size
    size = width_, height_
    for idx, model_name in enumerate(model_names):
        if idx < len(model_names) - 1: continue

        ret = dfprocessing_func([model_name, x_, x_val, x_test])
        print(ret)
        train_val, validation_val, test_val =  ret[0], ret[1], ret[2]

        validation_values.append(validation_val)
        train_values.append(train_val)
        test_values.append(test_val)
        # del x, y, x_val, y_val, x_test, y_test

        print('\n\n\n')



    for i in weights_name:
        print(i, end='\t')
        print(i, end='\t', file=log_file)
    print()
    print(file=log_file)
    for i in train_values:
        print(i, end='\t')
        print(i, end='\t', file=log_file)
    print()
    print(file=log_file)
    for i in validation_values:
        print(i, end='\t')
        print(i, end='\t', file=log_file)
    print()
    print(file=log_file)
    for i in test_values:
        print(i, end='\t')
        print(i, end='\t', file=log_file)
    print()
    print(file=log_file)
    log_file.flush()
    log_file.close()
