from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from lib.dataset.dataset import ClassificationDataset
from lib.metrics.metrics import compute_metrics
from lib.metrics.loss import BalancedTrainer
from lib.utils.utils import oversample_data, undersample_data, ratio_undersample
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model_id', default='albert-base-v2',help="name of the model to use")
    parser.add_argument('--num_epochs', default=3, help="number of training steps")
    parser.add_argument('--trainer_mode',default = 'default')
    parser.add_argument('--sampling_mode',default='none', help='can be "over" or "under" to balance out the dataset')
    parser.add_argument('--sub_frac',default=0.1, help="Fraction of the dataset to subsample")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """
    A simple training script for finetuning a pre-trained model
    """
    # get arguments
    args = get_args()
    model_id = args.model_id
    num_epochs = int(args.num_epochs)
    trainer_mode = args.trainer_mode
    sampling = args.sampling_mode
    # Subsample the training dataset to obtain a validation dataset
    train_csv = pd.read_csv('/home/ubuntu/adithya/temp/EAMLA/Natural Language Processing/atis_intents_train.csv', names=['label','text'])
    test_csv = pd.read_csv('/home/ubuntu/adithya/temp/EAMLA/Natural Language Processing/atis_intents_test.csv', names=['label','text'])
    train_csv, val_csv = train_test_split(train_csv, test_size = 0.1, random_state = 200)

    # get label mappings
    unique_labels = train_csv['label'].unique().tolist()
    label_mapping = {x:i for i,x in enumerate(unique_labels)}

    # check sampling approach
    if args.sampling_mode == 'over':
        print("Oversampling training data")
        train_csv = oversample_data(train_csv)
    elif args.sampling_mode == 'under':
        print("Undersampling training data")
        train_csv = undersample_data(train_csv)
    elif args.sampling_mode == 'frac':
        frac = float(args.sub_frac)
        train_csv = ratio_undersample(train_csv, frac)
    else:
        pass
    # load train and test datasets
    train_dataset = ClassificationDataset(train_csv,model_id=model_id,label_mapping=label_mapping)
    test_dataset = ClassificationDataset(test_csv,model_id=model_id,label_mapping=label_mapping)
    val_dataset = ClassificationDataset(val_csv,model_id=model_id,label_mapping=label_mapping)

    # get model
    model = AutoModelForSequenceClassification.from_pretrained(model_id,num_labels = len(train_dataset.unique_labels))

    # define trainer
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=int(len(train_dataset)/16),
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=5,
        load_best_model_at_end=True,
        no_cuda=False
    )
    print(args.trainer_mode)
    if args.trainer_mode == 'default':
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
    else:
        trainer = BalancedTrainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
    trainer.train()

    # evaluate
    test_results = trainer.evaluate(test_dataset)
    val_results = trainer.evaluate(val_dataset)
    print(f"Final val accuracy:{val_results['eval_accuracy']}")
    print(f"Final test accuracy:{test_results['eval_accuracy']}")
    print(f"Final test f1_score micro:{test_results['eval_f1_score_micro']}")
    print(f"Final test f1_score macro:{test_results['eval_f1_score_macro']}")
