# Supervised NLP
A simple training framework for text classification. Supports all huggingface transformer models.

## Installation
```
conda create -n ssn python=3.8
pip install -r requirements.txt
```

## Running Training
```
python intent_classsifier.py args
```
| ARGS | Desc |
| --------------- | --------------- |
| --model_id | Name of model. Refer to transformers package for more. default='albert' |
| --num_epochs | Number of epochs to train for |
| --trainer_mode | Use balanced loss trainer if not == 'default' |
| --sampling_mode | Over, under or numerically sample |
| --sub_frac | If subsampling dataset, fraction to use for training. |

Currently not integrated with wandb