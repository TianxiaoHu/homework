#!/usr/bin/env python
import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('model_save_path', type=str)
    parser.add_argument('epochs', type=int, default=10)

    args = parser.parse_args()

    print('loading expert data in ', args.expert_data_file)
    f = open(args.expert_data_file, 'rb')
    data = pickle.load(f)
    f.close

    x = data["observations"]
    y = data["actions"]
    print("observation shape", x.shape)
    print("actions shape", y.shape)

    y_shape = y.shape

    y = y.reshape(y_shape[0], int(np.product(y_shape) / y_shape[0]))
    print("actions reshaped to ", y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu,
                           input_shape=(x.shape[1], )),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(y.shape[1])
    ])

    model.compile(optimizer='adam', loss="mse", metrics=['mse'])
    model.fit(x_train, y_train, epochs=args.epochs)
    print("model score", model.evaluate(x_test, y_test))

    model.save(args.model_save_path)
    print("model saved to ", args.model_save_path)


if __name__ == '__main__':
    main()
