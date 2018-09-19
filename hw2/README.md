# Instructions

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

