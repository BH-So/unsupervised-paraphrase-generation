import argparse
import json
from datetime import datetime
import random
import logging

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from model.gpt2_trainer import FinetuneGPT2
from data.data_loader import QQPDataset


start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def train(args):
    device = args.device
    batch_size = args.batch_size

    gpt_model = FinetuneGPT2(args)
    gpt_model.build_model(checkpoint_dir=args.checkpoint)

    train_dataset = QQPDataset(gpt_model.tokenizer, args.train_data_path,
                               max_length=args.max_length,
                               load_augmented=True,
                               device=device, is_toy=args.toy)
    dev_dataset = QQPDataset(gpt_model.tokenizer, args.dev_data_path,
                             max_length=args.max_length,
                             device=device, is_toy=args.toy)

    logging.info("Start training")
    last_step = 0
    for begin_loc in range(0, len(train_dataset), batch_size):
        last_step += 1

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=300,  # warmup_steps=gpt_model.num_warmup_steps,
        weight_decay=0.01,
        evaluate_during_training=True,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        seed=args.seed,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=gpt_model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tb_writer=gpt_model.writer,
        prediction_loss_only=True,
    )
    # optimizers=(gpt_model.optimizer, gpt_model.scheduler),

    trainer.train()
    trainer.save_model()

    trainer.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str,
                        default='./data/QQP_split/train_preparation.txt',
                        help='train dataset file')
    parser.add_argument('--dev_data_path', type=str,
                        default='./data/QQP_split/dev_preparation.txt',
                        help='dev dataset file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to LOAD model checkpoint')
    parser.add_argument('--save_dir', type=str,
                        help='Path to SAVE model checkpoint')
    parser.add_argument('--summary_dir', type=str,
                        help='Path to save tensorboard summary')
    parser.add_argument('--log', type=str,
                        default='./logs/train_{datetime}.log',
                        help='Log filename')
    parser.add_argument('--device', type=str, default='cuda',
                        help='{cuda, cpu}')

    parser.add_argument('--model', type=str, default='gpt2-medium',
                        help='pretrained model name (only gpt available)')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum number of tokens for each sequence')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Evaluation batch size')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Number of update steps to accumulate the gradients')
    parser.add_argument('--learning_rate', type=float, default=6.25e-5,
                        help='Learning rate of fine-tuning')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.002,
                        help='Linear warmup ratio [0, 1)')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Number of update steps before eval & save')

    parser.add_argument('--tag', type=str,
                        help='Add a suffix of checkpoints')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    if args.save_dir is None:
        save_dir = args.model
        save_dir += '_toy' if args.toy else ''
        save_dir += '_{}'.format(args.tag) if args.tag else ''
        save_dir += '_continued' if args.checkpoint is not None else ''
        save_dir += '_{}'.format(start_datetime)
        args.save_dir = './checkpoints/{}/'.format(save_dir)

    if args.summary_dir is None:
        args.summary_dir = args.save_dir.replace('checkpoints', 'runs')

    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = args.log
    if args.tag:
        log_file = log_file.replace('{datetime}', args.tag + '_{datetime}')
    logging.basicConfig(level=log_level, format=log_format,
                        filename=log_file.format(datetime=start_datetime))
    logging.getLogger().setLevel(log_level)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.toy:
        args.train_data_path = args.dev_data_path
    logging.info('Parsed args: ' + json.dumps(dict(args.__dict__), indent=2))

    train(args)
