import numpy as np
import pickle as cp
import argparse

from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Conv2D, GlobalMaxPooling2D
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix

import reservoir
import utils
from time import time


# Instantiate the parser
parser = argparse.ArgumentParser(description='ConvESN_MSMC')

# Required positional argument
parser.add_argument('input_folder', default='./data/padded', help='the skeleton data folder name')
parser.add_argument('split_number', default='1', help='split number to consider')
parser.add_argument('--fit_model', action='store_true')

args = parser.parse_args()


print('Loading data...')
"""
a p file is a list: [left_hand_skeleton, right_hand_skeleton, left_leg_skeleton, right_leg_skeleton, central_trunk_skeleton, labels]
the shape of the first five ones: (num_samples, time_length, num_joints)
the shape of the last one: (num_samples,)
"""
# filepath_train = './dataset/MSRAction3D_real_world_P4_Split_AS3_train.p'
# filepath_test = './dataset/MSRAction3D_real_world_P4_Split_AS3_test.p'
filepath_train = args.input_folder + '/MSRAction3D_real_world_P4_Split_AS' + args.split_number + '_train.p'
filepath_test = args.input_folder + '/MSRAction3D_real_world_P4_Split_AS' + args.split_number + '_test.p'

data_train = cp.load(open(filepath_train, 'rb'))
skeletons_train = data_train[0:5]
labels_train = data_train[5]
data_test = cp.load(open(filepath_test, 'rb'))
skeletons_test = data_test[0:5]
labels_test = data_test[5]


def print_shapes(skeletons_data, annotation="train"):
    for skeleton in skeletons_data:
        print(annotation, "::::", skeleton.shape)


print_shapes(skeletons_train, "train")
print_shapes(skeletons_test, "test")

print('Transfering labels...')
labels_train, labels_test, num_classes = utils.transfer_labels(labels_train, labels_test)

"""
set parameters of reservoirs, create five reservoirs and get echo states of five skeleton parts
"""
num_samples_train = labels_train.shape[0]
num_samples_test = labels_test.shape[0]

_, time_length, n_in = skeletons_train[0].shape
n_res = n_in * 3
IS = 0.1
SR = 0.9
sparsity = 0.3
leakyrate = 1.0

# create five different reservoirs, one for a skeleton part
reservoirs = [reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity, leakyrate) for i in range(5)]

print('Getting echo states...')
echo_states_train = [np.empty((num_samples_train, 1, time_length, n_res), np.float32) for i in range(5)]
echo_states_test = [np.empty((num_samples_test, 1, time_length, n_res), np.float32) for i in range(5)]
for i in range(5):
    echo_states_train[i][:, 0, :, :] = reservoirs[i].get_echo_states(skeletons_train[i])
    echo_states_test[i][:, 0, :, :] = reservoirs[i].get_echo_states(skeletons_test[i])
echo_states_train = [np.concatenate(echo_states_train[0:2], axis=1), np.concatenate(echo_states_train[2:4], axis=1), echo_states_train[4]]
echo_states_test = [np.concatenate(echo_states_test[0:2], axis=1), np.concatenate(echo_states_test[2:4], axis=1), echo_states_test[4]]

"""
set parameters of convolution layers and build the MSSC decoder model
"""
input_shapes = ((2, time_length, n_res), (2, time_length, n_res), (1, time_length, n_res))

nb_filter = 16
nb_row = (2, 3, 4)  # time scales
nb_col = n_res
kernel_initializer = 'lecun_uniform'
activation = 'relu'
padding = 'valid'
strides = (1, 1)
data_format = 'channels_first'

optimizer = 'adam'
batch_size = 8
nb_epoch = 50
verbose = 1

# build the MSMC decoder model
inputs = []
features = []
for i in range(3):
    input = Input(shape=input_shapes[i])
    inputs.append(input)

    pools = []
    for j in range(len(nb_row)):
        conv = Conv2D(nb_filter, (nb_row[j], nb_col), kernel_initializer=kernel_initializer, activation=activation, padding=padding, strides=strides, data_format=data_format)(input)
        pool = GlobalMaxPooling2D(data_format=data_format)(conv)
        pools.append(pool)

    features.append(concatenate(pools))

"""
hands_features = features[0]
legs_features = features[1]
trunk_features = features[2]
body_features = Dense(nb_filter * len(nb_row), kernel_initializer = kernel_initializer, activation = activation)(concatenate([hands_features, legs_features, trunk_features]))
"""
body_features = Dense(nb_filter * len(nb_row), kernel_initializer=kernel_initializer, activation=activation)(concatenate(features))

outputs = Dense(num_classes, kernel_initializer=kernel_initializer, activation='softmax')(body_features)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

checkpoint_file = "check_points/weights-improvement_best.hdf5"

checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, tensorboard]

# model.fit(echo_states_train, labels_train, batch_size=batch_size, epochs=nb_epoch, verbose=verbose, validation_split=0.33, callbacks=callbacks_list)
if args.fit_model:
    model.fit(echo_states_train, labels_train, batch_size=batch_size, epochs=nb_epoch, verbose=verbose, validation_data=(echo_states_test, labels_test), callbacks=callbacks_list)

model.load_weights(checkpoint_file)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

scores = model.evaluate(echo_states_test, labels_test, batch_size=batch_size, verbose=verbose)
print("{}: {} and loss is {}".format(model.metrics_names[1], scores[1] * 100, scores[0]))

labels_test_pred = model.predict(echo_states_test)

print(labels_test.shape, labels_test_pred.shape)
print(confusion_matrix(labels_test, labels_test_pred))
