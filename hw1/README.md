# Instructions

## Setup

make new directories for different sections:

```bash
$ mkdir expert_data
$ mkdir bc_model
$ mkdir dagger_model
```

## Section 2

Take `Ant` as an example:

```bash
# get expert data
python run_expert.py experts/Ant-v2.pkl Ant-v2 --num_rollouts 20
# train behavior cloning model with expert data
python behavior_cloning.py expert_data/Ant-v2.pkl bc_model/Ant-v2.h5
# test behavior cloning model
python run_bc.py bc_model/Ant-v2.h5 Ant-v2 --num_rollouts 20 --render
```

Results are listed below:

|                 | expert_mean | expert_std | bc_mean | bc_std |
| --------------- | ----------- | ---------- | ------- | ------ |
| **Ant**         | 4831.49     | 121.63     | 4612.72 | 93.05  |
| **HalfCheetah** | 4141.69     | 65.94      | 4065.79 | 83.44  |
| Hopper          | 3779.38     | 4.38       | 781.79  | 117.69 |
| Humanoid        | 10393.55    | 40.76      | 346.83  | 62.93  |
| Reacher         | -3.78       | 1.83       | -7.11   | 1.82   |
| Walker2d        | 5514.88     | 67.71      | 445.85  | 427.00 |

(Note that task `Reacher` has different default `steps` with other tasks)

Afterwards I choose different numer of `epoches` as the hyperparameter.

Experiments are performed in task `Ant`.

```bash
# train behavior cloning model with different parameter `epochs`
# default is 10
# and evaluate different models
python behavior_cloning.py expert_data/Ant-v2.pkl bc_model/Ant-v2.h5 1
python run_bc.py bc_model/Ant-v2.h5 Ant-v2 --num_rollouts 20 --render
python behavior_cloning.py expert_data/Ant-v2.pkl bc_model/Ant-v2.h5 3
python run_bc.py bc_model/Ant-v2.h5 Ant-v2 --num_rollouts 20 --render
...
python behavior_cloning.py expert_data/Ant-v2.pkl bc_model/Ant-v2.h5 15
python run_bc.py bc_model/Ant-v2.h5 Ant-v2 --num_rollouts 20 --render
```
Results are listed below:

|      | 1       | 3       | 5       | 7       | 9       | 11      | 13      | 15      |
| ---- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| mean | 1265.84 | 3668.84 | 3961.06 | 4525.06 | 4634.40 | 4794.22 | 4680.76 | 4780.86 |
| std  | 590.00  | 1184.59 | 1288.10 | 77.69   | 111.45  | 126.88  | 78.30   | 95.73   |

## Section 3

Experiments are performed in task `Reacher`

```bash
# set different `iterations` and check the mean and std of rewards
python run_dagger.py experts/Reacher-v2.pkl expert_data/Reacher-v2.pkl dagger_model Reacher-v2 --num_rollouts=20 --iterations=1
python run_dagger.py experts/Reacher-v2.pkl expert_data/Reacher-v2.pkl dagger_model Reacher-v2 --num_rollouts=20 --iterations=3
...
python run_dagger.py experts/Reacher-v2.pkl expert_data/Reacher-v2.pkl dagger_model Reacher-v2 --num_rollouts=20 --iterations=15
```

Results are listed below:

|      | 1     | 3     | 5     | 7     | 9     | 11    | 13    | 15    |
| ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| mean | -7.70 | -5.32 | -5.10 | -4.64 | -5.00 | -4.43 | -3.71 | -3.33 |
| std  | 2.97  | 1.74  | 1.52  | 1.62  | 2.33  | 1.89  | 1.49  | 1.64  |

## Plot

Run a simple python script for generating plots.

2-2.eps and 3-2.eps will be automatically saved to current folder.

```bash
python plot.py
```

