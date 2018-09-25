import numpy as np
import pickle
import yaml
import sys
from time import time

from keras.models import Model, load_model
from keras.layers import Input, Dense, concatenate
from keras.layers import Conv2D, GlobalMaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint

import reservoir
import utils

total_reservoirs = 5


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


def predict_sample(args):
    print("Setting Up")
    skeletons, _ = get_data(args['input'])
    _, time_length, n_in = skeletons[0].shape
    with open(args['reservoir_file'], 'rb') as f:
        reservoirs = pickle.load(f)
    echo_states_test = [np.empty((1, 1, time_length, n_in * 3), np.float32) for i in range(5)]
    for i in range(total_reservoirs):
        echo_states_test[i][:, 0, :, :] = reservoirs[i].get_echo_states(skeletons[i])
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

    n_res = n_in * args["expansion_factor"]

    reservoirs = []
    if args["train"]:
        # create five different reservoirs, one for a skeleton part
        reservoirs = [reservoir.reservoir_layer(n_in, n_res, args["IS"], args["SR"], args["sparsity"], args["leakyrate"]) for i in range(total_reservoirs)]
        with open(args["reservoir_file"], 'wb') as f:
            pickle.dump(reservoirs, f)
    else:
        with open(args["reservoir_file"], 'rb') as f:
            reservoirs = pickle.load(f)
            n_res = reservoirs[0].n_res

    if args["verbose"]:
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

    if args["train"]:

        # build the MSMC decoder model
        inputs = []
        features = []
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
        body_features = Dense(args["nb_filter"] * len(args["nb_row"]), kernel_initializer = kernel_initializer, activation = activation)(concatenate([hands_features, legs_features, trunk_features]))
        """
        body_features = Dense(args["nb_filter"] * len(args["nb_row"]), kernel_initializer=args["kernel_initializer"], activation=args["activation"])(concatenate(features))

        outputs = Dense(num_classes, kernel_initializer=args["kernel_initializer"], activation='softmax')(body_features)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=args["optimizer"], loss='categorical_crossentropy', metrics=['accuracy'])

        tensorboard = TensorBoard(log_dir="logs/test_{}".format(time()), histogram_freq=0, batch_size=args["batch_size"], write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

        checkpoint = ModelCheckpoint(args["checkpoint_file"], monitor='val_acc', verbose=args["verbose"], save_best_only=True, mode='max')
        callbacks_list = [checkpoint, tensorboard]

        model.fit(echo_states_train, labels_train, batch_size=args["batch_size"], epochs=args["nb_epochs"], verbose=args["verbose"], validation_data=(echo_states_test, labels_test), callbacks=callbacks_list)

    try:
        model = load_model(args["checkpoint_file"])
    except OSError as err:
        print("OS error: {0}".format(err))
        sys.exit(1)

    print("==Evaluating==")
    scores = model.evaluate(echo_states_test, labels_test, batch_size=args["batch_size"], verbose=args["verbose"])
    print("{}: {} and loss is {}".format(model.metrics_names[1], scores[1] * 100, scores[0]))

    labels_test_pred = model.predict(echo_states_test)

    print("parameters :::", model.count_params())
    # print("summary :::", model.summary())
    # print(confusion_matrix(labels_test, labels_test_pred))


def main():
    # load config file
    if sys.argv[1]:
        with open(sys.argv[1]) as f:
            args = yaml.safe_load(f)
    else:
        print("missing param :: Config File")
        return

    if args['test_sample']:
        predict_sample(args)
    else:
        run_MSR_MSMC(args)


if __name__ == "__main__":
    main()
