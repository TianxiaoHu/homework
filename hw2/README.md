# Instructions

This file contains all command-line expressions I used to run my experiments.

## Problem 4

```bash
// for experiments
$ python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
$ python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
$ python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
$ python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
$ python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
$ python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na

// generate plot for experiments prefixed with sb_
$ python plot.py data/sb_no_rtg_dna_CartPole-v0_18-09-2018_16-14-43/ data/sb_rtg_dna_CartPole-v0_18-09-2018_16-16-33/ data/sb_rtg_na_CartPole-v0_18-09-2018_16-25-17/

// generate plot for experiments prefixed with lb_
$ python plot.py data/lb_no_rtg_dna_CartPole-v0_18-09-2018_16-29-33/ data/lb_rtg_dna_CartPole-v0_18-09-2018_16-38-42/ data/lb_rtg_na_CartPole-v0_18-09-2018_16-50-27/
```

## Problem 5

```bash
// set parameter `batchsize` = 64 and `learning rate` = 6e-3
$ python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b 64 -lr 6e-3 -rtg --exp_name hc_b64_r6e-3

// generate plot
$ python plot.py data/hc_b64_r6e-3_InvertedPendulum-v2_18-09-2018_20-33-52/ --value AverageReturn
```

## Problem 7

```bash
// setup the enviornment
// make sure to copy the provided file `lunar_lander.py` into the path gym/envs/box2d/and replace the original file `lunar_lander.py`
// install box2d and its dependency
$ brew install swig // you may need to update homebrew first
$ pip install box2d box2d-kengz

$ python train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005

// generate plot
$ python plot.py data/ll_b40000_r0.005_LunarLanderContinuous-v2_18-09-2018_21-56-17/ --value AverageReturn
```

## Problem 8

```bash
// search for the best set of parameters
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.005 -rtg --nn_baseline --exp_name hc_b10000_r0.005
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.01 -rtg --nn_baseline --exp_name hc_b10000_r0.01
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.02 -rtg --nn_baseline --exp_name hc_b10000_r0.02
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.005 -rtg --nn_baseline --exp_name hc_b30000_r0.005
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.01 -rtg --nn_baseline --exp_name hc_b30000_r0.01
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 -rtg --nn_baseline --exp_name hc_b30000_r0.02
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.005 -rtg --nn_baseline --exp_name hc_b50000_r0.005
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.01 -rtg --nn_baseline --exp_name hc_b50000_r0.01
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name hc_b50000_r0.02

// best parameter set is [50000, 0,02]
// remember to change the --exp_name parameter or plot could not be generated
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 --exp_name hc_b50000_r0.02
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --exp_name hc_rtg_b50000_r0.02
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 --nn_baseline --exp_name hc_nn_b50000_r0.02
$ python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name hc_rtg_nn_b50000_r0.02

// generate plot
$ python plot.py data/hc_b50000_r0.02_HalfCheetah-v2_19-09-2018_01-24-22/ data/hc_b50000_r0.02_HalfCheetah-v2_19-09-2018_02-28-23/ data/hc_b50000_r0.02_HalfCheetah-v2_19-09-2018_03-56-51/ data/hc_b50000_r0.02_HalfCheetah-v2_19-09-2018_04-56-16/
```

