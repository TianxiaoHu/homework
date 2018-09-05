#!/usr/bin/env python
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from tensorflow import keras


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('behavioral_cloning_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading behavioral cloning model')
    model = keras.models.load_model(args.behavioral_cloning_file)
    print('successfully loaded')

    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        if i % 10 == 0:
            print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model.predict(obs.reshape(1, obs.shape[0]))
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))


if __name__ == '__main__':
    main()
