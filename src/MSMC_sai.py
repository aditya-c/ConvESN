import numpy as np
import pickle
import argparse
import sys
from time import time

from keras.models import Model, load_model
from keras.layers import Input, Dense, concatenate
from keras.layers import Conv2D, GlobalMaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint

import reservoir
import utils


def parse_args():
    """
    returns {input, split_number, train}
    """
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='ConvESN_MSMC')

    # location of the padded skeleton data
    parser.add_argument('input', default='./data/padded', help='the skeleton data folder/ test file name')
    # choose the split number from the padded files
    parser.add_argument('-split_number', nargs='?', default='1', help='split number to use')
    # trains a model if set to true
    parser.add_argument('--train', action='store_true')
    # name of checkpoint file (save to/ load from)
    parser.add_argument('-checkpoint', default='check_points/weights-improvement_test.hdf5', nargs='?', help="name of checkpoint file to load/save")
    # save reservoir along with model
    parser.add_argument('-reservoir', default='reservoir/reservoir_test.pkl', nargs='?', help="name of checkpoint file to load/save")
    parser.add_argument('-test_sample', action='store_true')

    return parser.parse_args()


def get_data(filename):
    """
    returns {skeleton data, labels}

    skeleton pickle file is a list: [left_hand_skeleton, right_hand_skeleton, left_leg_skeleton, right_leg_skeleton, central_trunk_skeleton, labels]
    the shape of the first five ones: (num_samples, time_length, num_joints)
    the shape of the last one: (num_samples,)
    """
    data = pickle.load(open(filename, 'rb'))
    skeletons, labels = data[0:5], data[5]
    return skeletons, labels


def print_shapes(skeletons_data, annotation="train"):
    for skeleton in skeletons_data:
        print(annotation, "::::", skeleton.shape)


if __name__ == "__main__":
    total_reservoirs = 5
    # parse arguments
    args = parse_args()

    if args.test_sample:
        print("Setting Up")
        skeletons, _ = get_data(args.input)
        _, time_length, n_in = skeletons[0].shape
        with open(args.reservoir, 'rb') as f:
            reservoirs = pickle.load(f)
        echo_states_test = [np.empty((1, 1, time_length, n_in * 3), np.float32) for i in range(5)]
        for i in range(total_reservoirs):
            echo_states_test[i][:, 0, :, :] = reservoirs[i].get_echo_states(skeletons[i])
        echo_states_test = [np.concatenate(echo_states_test[0:2], axis=1), np.concatenate(echo_states_test[2:4], axis=1), echo_states_test[4]]
        model = load_model(args.checkpoint)
        print(f"Action :::: {model.predict(echo_states_test)}")

    # for multiple files

    # filepath_train = './dataset/MSRAction3D_real_world_P4_Split_AS3_train.p'
    # filepath_test = './dataset/MSRAction3D_real_world_P4_Split_AS3_test.p'
    filepath_train = args.input + '/MSRAction3D_real_world_P4_Split_AS' + args.split_number + '_train.p'
    filepath_test = args.input + '/MSRAction3D_real_world_P4_Split_AS' + args.split_number + '_test.p'

    # load data
    skeletons_train, labels_train = get_data(filepath_train)
    skeletons_test, labels_test = get_data(filepath_test)

    # print shapes
    # print_shapes(skeletons_train, "train")
    # print_shapes(skeletons_test, "test")

    # one hot of labels
    print('Transfering labels...')
    labels_train, labels_test, num_classes = utils.transfer_labels(labels_train, labels_test)

    """
    set parameters of reservoirs, create five reservoirs and get echo states of five skeleton parts
    """
    num_samples_train = labels_train.shape[0]
    num_samples_test = labels_test.shape[0]

    _, time_length, n_in = skeletons_train[0].shape
    n_res = n_in
    IS = 0.1
    SR = 0.9
    sparsity = 0.3
    leakyrate = 1.0

    reservoirs = []
    if args.train:
        # create five different reservoirs, one for a skeleton part
        reservoirs = [reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity, leakyrate) for i in range(total_reservoirs)]
        with open(args.reservoir, 'wb') as f:
            pickle.dump(reservoirs, f)
    else:
        with open(args.reservoir, 'rb') as f:
            reservoirs = pickle.load(f)

    print('Getting echo states...')
    echo_states_train = [np.empty((num_samples_train, 1, time_length, n_res), np.float32) for i in range(5)]
    echo_states_test = [np.empty((num_samples_test, 1, time_length, n_res), np.float32) for i in range(5)]
    for i in range(total_reservoirs):
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
    nb_epoch = 100
    verbose = 1

    if args.train:

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

        tensorboard = TensorBoard(log_dir="logs/test_{}".format(time()), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

        checkpoint = ModelCheckpoint(args.checkpoint, monitor='val_acc', verbose=verbose, save_best_only=True, mode='max')
        callbacks_list = [checkpoint, tensorboard]

        model.fit(echo_states_train, labels_train, batch_size=batch_size, epochs=nb_epoch, verbose=verbose, validation_data=(echo_states_test, labels_test), callbacks=callbacks_list)

    try:
        model = load_model(args.checkpoint)
    except OSError as err:
        print("OS error: {0}".format(err))
        sys.exit(1)

    print("==Evaluating==")
    scores = model.evaluate(echo_states_test, labels_test, batch_size=batch_size, verbose=verbose)
    print("{}: {} and loss is {}".format(model.metrics_names[1], scores[1] * 100, scores[0]))

    labels_test_pred = model.predict(echo_states_test)

    print(labels_test.shape, labels_test_pred.shape)
    print("parameters :::", model.count_params())
    # print("summary :::", model.summary())
    # print(confusion_matrix(labels_test, labels_test_pred))

    with open("results.txt", "a+") as f:
        print("With reservoir", file=f)
        print("nb_epoch : {}, optimizer : {}, n_res : {}".format(nb_epoch, optimizer, n_res), file=f)
        print("{}: {} and loss is {}".format(model.metrics_names[1], scores[1] * 100, scores[0]), file=f)
        print("parameters :::", model.count_params(), file=f)
        print("*" * 15, file=f)
