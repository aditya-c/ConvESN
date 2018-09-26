import numpy as np
import pickle
import yaml
import sys
from datetime import datetime

from keras.models import Model, load_model
from keras.layers import Input, Dense, concatenate
from keras.layers import Conv2D, GlobalMaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint

import reservoir
import utils

skeleton_parts = 5


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
    """
    print shape of list of np objects
    ex: print_shapes(skeletons_test, "test")
    """
    for skeleton in skeletons_data:
        print(annotation, "::::", skeleton.shape)


def get_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def predict_sample(args):
    print("Setting Up")
    skeletons, _ = get_data(args['input'])
    _, time_length, n_in = skeletons[0].shape
    with open(args['reservoir_file'], 'rb') as f:
        reservoirs = pickle.load(f)
    reservoir_nums = [0, 0, 1, 1, 2] if len(reservoirs) != skeleton_parts else list(range(skeleton_parts))

    echo_states_test = [np.empty((1, 1, time_length, n_in * 3), np.float32) for i in range(skeleton_parts)]
    for i, reservoir_num in enumerate(reservoir_nums):
        echo_states_test[i][:, 0, :, :] = reservoirs[reservoir_num].get_echo_states(skeletons[i])
    echo_states_test = [np.concatenate(echo_states_test[0:2], axis=1), np.concatenate(echo_states_test[2:4], axis=1), echo_states_test[4]]
    model = load_model(args["checkpoint_file"])
    print(f"Action :::: {model.predict(echo_states_test)}")


def run_MSR_MSMC(args):
    # load data
    skeletons_train, labels_train = get_data(args["input_train_file"])
    skeletons_test, labels_test = get_data(args["input_test_file"])

    # one hot of labels
    labels_train, labels_test, num_classes = utils.transfer_labels(labels_train, labels_test)

    """
    set parameters of reservoirs, create five reservoirs and get echo states of five skeleton parts
    """
    num_samples_train, num_samples_test = labels_train.shape[0], labels_test.shape[0]

    _, time_length, n_in = skeletons_train[0].shape

    if args["use_ESN"]:
        n_res = n_in * args["expansion_factor"]

        # GET RESERVOIRS ####
        reservoirs = []
        if args["train"]:
            if args["common_reservoir_for_limbs"]:
                # create three different reservoirs
                reservoirs = [reservoir.reservoir_layer(n_in, n_res, args["IS"], args["SR"], args["sparsity"], args["leakyrate"]) for i in range(3)]
                # left_hand, right_hand, left_leg, right_leg, trunk
            else:
                # create five different reservoirs, one for a skeleton part - 5 parts
                reservoirs = [reservoir.reservoir_layer(n_in, n_res, args["IS"], args["SR"], args["sparsity"], args["leakyrate"]) for i in range(skeleton_parts)]
            with open(args["reservoir_file"], 'wb') as f:
                pickle.dump(reservoirs, f)
        else:
            with open(args["reservoir_file"], 'rb') as f:
                reservoirs = pickle.load(f)
                n_res = reservoirs[0].n_res

        reservoir_nums = [0, 0, 1, 1, 2] if len(reservoirs) != skeleton_parts else list(range(skeleton_parts))

        # GET ECHO STATES for the skeletons ####
        echo_states_train = [np.empty((num_samples_train, 1, time_length, n_res), np.float32) for i in range(skeleton_parts)]
        echo_states_test = [np.empty((num_samples_test, 1, time_length, n_res), np.float32) for i in range(skeleton_parts)]
        for i, reservoir_num in enumerate(reservoir_nums):
            echo_states_train[i][:, 0, :, :] = reservoirs[reservoir_num].get_echo_states(skeletons_train[i])
            echo_states_test[i][:, 0, :, :] = reservoirs[reservoir_num].get_echo_states(skeletons_test[i])
        echo_states_train = [np.concatenate(echo_states_train[0:2], axis=1), np.concatenate(echo_states_train[2:4], axis=1), echo_states_train[4]]
        echo_states_test = [np.concatenate(echo_states_test[0:2], axis=1), np.concatenate(echo_states_test[2:4], axis=1), echo_states_test[4]]

        input_train, input_test = echo_states_train, echo_states_test

    else:
        n_res = n_in
        skeletons_train_ = [np.expand_dims(x, 1) for x in skeletons_train]
        skeletons_test_ = [np.expand_dims(x, 1) for x in skeletons_test]

        skeletons_train_ = [np.concatenate(skeletons_train_[0:2], axis=1), np.concatenate(skeletons_train_[2:4], axis=1), skeletons_train_[4]]
        skeletons_test_ = [np.concatenate(skeletons_test_[0:2], axis=1), np.concatenate(skeletons_test_[2:4], axis=1), skeletons_test_[4]]

        input_train, input_test = skeletons_train_, skeletons_test_

    # TRAIN MSMC MODEL ####
    if args["train"]:

        input_shapes = ((2, time_length, n_res), (2, time_length, n_res), (1, time_length, n_res))
        inputs = []
        features = []

        # BUILD THE MSMC decoder model ####
        for i in range(3):
            input = Input(shape=input_shapes[i])
            inputs.append(input)

            pools = []
            for j in range(len(args["nb_row"])):
                conv = Conv2D(args["nb_filter"], (args["nb_row"][j], n_res), kernel_initializer=args["kernel_initializer"], activation=args["activation"], padding=args["padding"], strides=args["strides"], data_format=args["data_format"])(input)
                pool = GlobalMaxPooling2D(data_format=args["data_format"])(conv)
                pools.append(pool)

            features.append(concatenate(pools))

        """
        hands_features = features[0]
        legs_features = features[1]
        trunk_features = features[2]
        """
        body_features = Dense(args["nb_filter"] * len(args["nb_row"]), kernel_initializer=args["kernel_initializer"], activation=args["activation"])(concatenate(features))

        outputs = Dense(num_classes, kernel_initializer=args["kernel_initializer"], activation='softmax')(body_features)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=args["optimizer"], loss='categorical_crossentropy', metrics=['accuracy'])

        # CALLBACKS ####
        log_file = args["log_dir"] + "/{}_res{}_com{}".format(get_time(), args["use_ESN"], args["common_reservoir_for_limbs"])
        tensorboard = TensorBoard(log_dir=log_file, histogram_freq=0, batch_size=args["batch_size"], write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        checkpoint = ModelCheckpoint(args["checkpoint_file"], monitor='val_acc', verbose=args["verbose"], save_best_only=True, mode='max')
        callbacks_list = [checkpoint, tensorboard]

        # FIT MODEL ####
        model.fit(input_train, labels_train, batch_size=args["batch_size"], epochs=args["nb_epochs"], verbose=args["verbose"], validation_data=(input_test, labels_test), callbacks=callbacks_list)

    # LOAD BEST MODEL ####
    try:
        model = load_model(args["checkpoint_file"])
    except OSError as err:
        print("OS error: {0}".format(err))
        return

    # EVALUATE MODEL ####
    print("==Evaluating==")
    scores = model.evaluate(input_test, labels_test, batch_size=args["batch_size"], verbose=args["verbose"])
    print("{}: {} and loss is {}".format(model.metrics_names[1], scores[1] * 100, scores[0]))

    labels_test_pred = model.predict(input_test)

    print("parameters :::", model.count_params())
    # print("summary :::", model.summary())
    # print(confusion_matrix(labels_test, labels_test_pred))

    # SAVE RANDOM STUFF ####
    with open(args["results_file"], "a+") as f:
        print("Reservoir :: {} @ {}".format(args["use_ESN"], get_time()), file=f)
        if args["use_ESN"]:
            print("Common Limb Reservoir :: {}".format(args["common_reservoir_for_limbs"]), file=f)
        print("nb_epoch : {}, optimizer : {}, n_res : {}".format(args["nb_epochs"], args["optimizer"], n_res), file=f)
        print("{}: {} and loss is {}".format(model.metrics_names[1], scores[1] * 100, scores[0]), file=f)
        print("parameters :::", model.count_params(), file=f)
        print("*" * 15, file=f)


def MSMC(config_file):
    # load config file
    with open(config_file) as f:
        args = yaml.safe_load(f)

    if args['test_sample']:
        predict_sample(args)
    else:
        run_MSR_MSMC(args)


if __name__ == "__main__":
    if sys.argv[1]:
        MSMC(sys.argv[1])
    else:
        print("missing param :: Config File")
