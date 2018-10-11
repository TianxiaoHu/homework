# Instructions

This file contains all command-line expressions I used to run my experiments.

## Q-Learning

```bash
$ python run_dqn_atari.py // for vanilla Q-learning, set the `double_q` = False in the line 78 of run_dqn_atari.py
$ python run_dqn_atari.py // for double Q-learning, set the `double_q` = True in the line 78 of run_dqn_atari.py
$ python run_dqn_lander.py // for different learning rate, go to run_dqn_lander.py and set the line 26 `lr_schedule = ConstantSchedule(<lr>)` 

// plots about vanilla Q-learning, double Q-learning and hyperparamter is in plot.ipynb
```

## Actor-Critic

```bash
// run experiment on CartPole task
$ python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_1 -ntu 1 -ngsptu 1
$ python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 100_1 -ntu 100 -ngsptu 1
$ python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_100 -ntu 1 -ngsptu 100
$ python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 10_10 -ntu 10 -ngsptu 10

// generate plots and find best hyperparameter set
$ python plot.py data/ac_1_100_CartPole-v0_07-10-2018_16-04-51/ data/ac_10_10_CartPole-v0_07-10-2018_16-07-14/ data/ac_100_1_CartPole-v0_07-10-2018_16-02-02/

// train on more difficult tasks using hyperparameter set (10, 10)
$ python train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name 10_10 -ntu 10 -ngsptu 10
$ python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --exp_name 10_10 -ntu 10 -ngsptu 10

// generate plots
$ python plot.py data/ac_10_10_InvertedPendulum-v2_07-10-2018_16-11-53/
$ python plot.py data/ac_10_10_HalfCheetah-v2_07-10-2018_16-25-02/
```

