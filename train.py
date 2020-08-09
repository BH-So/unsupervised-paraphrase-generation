import argparse
import json
from datetime import datetime
import random
import logging

import numpy as np
import torch
from tqdm import tqdm

from model.gpt2_finetune_model import FinetuneGPT2
from data.data_loader import QQPDataset


start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def train(args):
    if args.save_dir is None:
        save_dir = args.model
        save_dir += '_toy' if args.toy else ''
        save_dir += '_{}'.format(args.tag) if args.tag else ''
        save_dir += '_continued' if args.checkpoint is not None else ''
        save_dir += '_{}'.format(start_datetime)
        args.save_dir = './checkpoints/{}/'.format(save_dir)
    if args.summary_dir is None:
        args.summary_dir = args.save_dir.replace('checkpoints', 'runs')
    device = args.device
    batch_size = args.batch_size

    gpt_model = FinetuneGPT2(args)

    if args.checkpoint is not None:
        gpt_model.load_saved_model(args.checkpoint)
    gpt_model.build_model()

    train_dataset = QQPDataset(gpt_model.tokenizer, args.train_data_path,
                               max_length=args.max_length,
                               device=device, is_toy=args.toy)
    dev_dataset = QQPDataset(gpt_model.tokenizer, args.dev_data_path,
                             max_length=args.max_length,
                             device=device, is_toy=args.toy)

    gpt_model.build_optimizer(dataset_size=len(train_dataset))

    eos_tok = gpt_model.tokenizer.eos_token
    sep_tok = gpt_model.tokenizer.sep_token
    dev_samples = [gpt_model.tokenizer.decode(ids).split(sep_tok)[0]
                   for ids in dev_dataset[:5]['input_ids']]
    train_samples = [gpt_model.tokenizer.decode(ids).split(sep_tok)[0]
                     for ids in train_dataset[:5]['input_ids']]

    logging.info("Start training")
    last_step = 0
    for begin_loc in range(0, len(train_dataset), batch_size):
        last_step += 1

    for ep in range(1, args.num_epochs+1):
        step = 0
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        for begin_loc in tqdm(range(0, len(train_dataset), batch_size)):
            step += 1
            end_loc = begin_loc + batch_size
            samples = train_dataset[indices[begin_loc:end_loc]]
            _ = gpt_model.train(samples)

            if step % 20000 == 0 or step == last_step:
            #        and (args.toy is False or ep % 200 == 0):
                if step < last_step:
                    logging.info("{} step of {} epoch".format(step, ep))
                    checkpoint = 'ep{}_{}k_steps/'.format(ep, step // 1000)
                else:
                    logging.info("End of {} epoch".format(ep))
                    checkpoint = 'ep{}/'.format(ep)
                if args.toy is False:
                    gpt_model.save_model(args.save_dir + checkpoint)
                ppl = gpt_model.get_loss(dev_dataset)
                print("Dev ppl: {}".format(ppl))
                print("Sample generation on train data")
                generated_texts = gpt_model.generate_text(
                        train_samples, suffix=sep_tok, eos=eos_tok)
                for inp, gen in zip(train_samples, generated_texts):
                    print("  Input: {}".format(inp))
                    print("  Gen: {}".format(gen))
                print("Sample generation on dev data")
                generated_texts = gpt_model.generate_text(
                        dev_samples, suffix=sep_tok, eos=eos_tok)
                for inp, gen in zip(dev_samples, generated_texts):
                    print("  Input: {}".format(inp))
                    print("  Gen: {}".format(gen))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str,
                        default='./data/QQP_split/train_preparation.txt',
                        help='train dataset file')
    parser.add_argument('--dev_data_path', type=str,
                        default='./data/QQP_split/dev_preparation.txt',
                        help='dev dataset file')
    parser.add_argument('--checkpoint', type=str,
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
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum number of tokens for each sequence')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=6.25e-5,
                        help='Learning rate of fine-tuning')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.002,
                        help='Linear warmup ratio [0, 1)')

    parser.add_argument('--tag', type=str,
                        help='Add a suffix of checkpoints')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = args.log
    if args.tag:
        log_file = log_file.replace('{datetime}', args.tag + '_{datetime}')
    logging.basicConfig(level=log_level, format=log_format,
                        filename=log_file.format(datetime=start_datetime))
    logging.getLogger().setLevel(log_level)

    ### Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ### Deterministic option may have a negative performance impact
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if args.toy:
        args.train_data_path = args.dev_data_path
    logging.info('Parsed args: ' + json.dumps(dict(args.__dict__), indent=2))

    train(args)
