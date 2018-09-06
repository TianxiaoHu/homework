#!/usr/bin/env python
import os
import tf_util
import gym
import load_policy
import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('model_save_path', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    print('loading expert data in ', args.expert_data_file)
    f = open(args.expert_data_file, 'rb')
    data = pickle.load(f)
    f.close()

    x = data["observations"]
    y = data["actions"]
    print("observation shape", x.shape)
    print("actions shape", y.shape)

    y_shape = y.shape

    y = y.reshape(y_shape[0], int(np.product(y_shape) / y_shape[0]))
    print("actions reshaped to ", y.shape)

    # Set Parameters
    mean_rewards = []
    stds = []

    for i in range(args.iterations):
        # (1) Train policy on D
        model = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu,
                               input_shape=(x.shape[1], )),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(y.shape[1])
        ])

        model.compile(optimizer='adam', loss="mse", metrics=['mse'])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3)
        model.fit(x_train, y_train, epochs=10)
        print("model score", model.evaluate(x_test, y_test))
        model.save(os.path.join(args.model_save_path,
                                args.envname + "-%i.h5" % (i)))

        # (2) Run policy on simulation
        # and
        # (3) Expert labels on these observations
        with tf.Session():
            tf_util.initialize()
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            new_observations = []
            new_exp_actions = []

            model = keras.models.load_model(os.path.join(
                args.model_save_path, args.envname + "-%i.h5" % (i)))
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    exp_action = policy_fn(obs[None, :])
                    action = (model.predict(obs.reshape(1, obs.shape[0])))

                    new_observations.append(obs)
                    new_exp_actions.append(exp_action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0:
                        print("%i/%i" % (steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
            mean_rewards.append(np.mean(returns))
            stds.append(np.std(returns))

            new_observations = np.array(new_observations)
            new_exp_actions = np.array(new_exp_actions)

        # (4) Aggregate new data to old
        new_observations = new_observations.reshape(
            (new_observations.shape[0], x.shape[1]))
        new_exp_actions = new_exp_actions.reshape(
            (new_exp_actions.shape[0], y.shape[1]))

        x = np.concatenate((x, new_observations))
        y = np.concatenate((y, new_exp_actions))

    print("rewards ", mean_rewards)
    print("std ", stds)


if __name__ == '__main__':
    main()
