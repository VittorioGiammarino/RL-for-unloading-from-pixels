# RL-for-unloading-from-pixels

## Initial instructions

**Step 1.** install miniconda

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Step 2.** clone repo and create conda environment

```shell
git clone GitHub Repository
```

```shell
cd GitHub Repository
conda env create -f environment.yml
conda activate RL4U
```

**Step 3.** Download the environment assets within the GitHub Repository

Assets: https://drive.google.com/file/d/1UqU3PPLOr9Y4cY9mQNXKK4QkxlnVVVlB/view?usp=sharing

or 

```shell
pip install gdown
gdown 1UqU3PPLOr9Y4cY9mQNXKK4QkxlnVVVlB
tar -xf assets.tar.xz
rm assets.tar.xz
```

**Step 4.** Test environment

To test the installation run the following command

```shell
python test_scripted_policy.py num_episodes=1 GUI=true
```

**Step 5.** Run experiments

Run experiments for Mask-off

```shell
python train_RL.py safety_mask=false reward_id=0 batch_size=64
```

Run experiments for Mask-off, v-reward

```shell
python train_RL.py safety_mask=false reward_id=1 batch_size=64
```

Run experiments for Mask-on

```shell
python train_RL.py safety_mask=true reward_id=0 batch_size=64
```

Run experiments for Mask-on, v-reward

```shell
python train_RL.py safety_mask=true reward_id=1 batch_size=64
```

### Troubleshooting

set batch_size=16 if CUDA memory error