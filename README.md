# PyTorch NADE and Orderless NADE

These are my PyTorch implementations of [NADE](http://proceedings.mlr.press/v15/larochelle11a.html) and [orderless NADE](https://arxiv.org/abs/1310.1757).

## Setting up `.nade_profile`

After you've cloned the repository to your desired location, create a file called `.nade_profile` in your home directory:

```bash
nano ~/.nade_profile
```

and copy and paste in the contents of [`.nade_profile`](.nade_profile), replacing each of the variable values with paths relevant to your environment.
Next, add the following line to the end of your `~/.bashrc`:

```bash
source ~/.nade_profile
```

and either log out and log back in again or run:

```bash
source ~/.bashrc
```

You should now be able to copy and paste all of the commands in the various instructions sections.
For example:

```bash
echo ${NADE_PROJECT_DIR}
```

should print the path you set for `NADE_PROJECT_DIR` in `.nade_profile`.

## Training NADE

Run (or copy and paste) the following script, editing the variables as appropriate.

```bash
#!/usr/bin/env bash

JOB=$(date +%Y%m%d%H%M%S)

echo "train:" >> ${JOB}.yaml
echo "  train_prop: 0.98" >> ${JOB}.yaml
echo "  epochs: 500" >> ${JOB}.yaml
echo "  batch_size: 1000" >> ${JOB}.yaml
echo "  workers: 10" >> ${JOB}.yaml
echo "  optimizer: adam" >> ${JOB}.yaml
echo "  learning_rate: 5.0e-3" >> ${JOB}.yaml
echo "  patience: 20" >> ${JOB}.yaml

echo "model:" >> ${JOB}.yaml
echo "  hidden_dim: 500" >> ${JOB}.yaml

# Save experiment settings.
mkdir -p ${NADE_EXPERIMENTS_DIR}/${JOB}
mv ${JOB}.yaml ${NADE_EXPERIMENTS_DIR}/${JOB}/

gpu=0
cd ${NADE_PROJECT_DIR}
nohup python3 train_nade.py ${JOB} ${gpu} > ${NADE_EXPERIMENTS_DIR}/${JOB}/train.log &
```

## Training Orderless NADE

Run (or copy and paste) the following script, editing the variables as appropriate.

```bash
#!/usr/bin/env bash

JOB=$(date +%Y%m%d%H%M%S)

echo "train:" >> ${JOB}.yaml
echo "  train_prop: 0.98" >> ${JOB}.yaml
echo "  epochs: 4000" >> ${JOB}.yaml
echo "  batch_size: 1000" >> ${JOB}.yaml
echo "  workers: 10" >> ${JOB}.yaml
echo "  learning_rate: 1.0e-3" >> ${JOB}.yaml
echo "  patience: 20" >> ${JOB}.yaml

echo "model:" >> ${JOB}.yaml
echo "  mlp_layers: [500, 500]" >> ${JOB}.yaml

# Save experiment settings.
mkdir -p ${NADE_EXPERIMENTS_DIR}/${JOB}
mv ${JOB}.yaml ${NADE_EXPERIMENTS_DIR}/${JOB}/

gpu=0
cd ${NADE_PROJECT_DIR}
nohup python3 train_orderless_nade.py ${JOB} ${gpu} > ${NADE_EXPERIMENTS_DIR}/${JOB}/train.log &
```
