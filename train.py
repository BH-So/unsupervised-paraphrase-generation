import argparse
from datetime import datetime
import os
import random
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
from data.data_loader import QQPDataset

import logging

device = 'cuda'


def get_ppl(model, dataset, batch_size=32):
    _ = model.eval()
    batch_size = 1
    with torch.no_grad():
        lls = []
        for begin_loc in range(0, len(dataset), batch_size):
            end_loc = begin_loc + batch_size
            samples = dataset[begin_loc:end_loc]
            outputs = model(**samples)
            log_likelihood = outputs[0]
            lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / len(dataset))
    _ = model.train()
    return ppl.item()


def train(args, model_name, train_filename, dev_filename, is_toy=False):
    num_epochs = 5 if is_toy == False else 1
    batch_size = 4
    eval_batch_size = 4

    if args.checkpoint is None:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        special_tokens_dict = {'sep_token': '[SEP]'}
        tokenizer.pad_token = tokenizer.eos_token
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.checkpoint)
        model = GPT2LMHeadModel.from_pretrained(args.checkpoint)

    model = model.to(device)
    _ = model.train()

    if is_toy:
        train_filename = dev_filename
    train_dataset = QQPDataset(tokenizer, train_filename,
                               device=device, is_toy=is_toy)
    dev_dataset = QQPDataset(tokenizer, dev_filename,
                             device=device, is_toy=is_toy)
    num_data = len(train_dataset)
    num_train_steps = ((num_data - 1) // batch_size + 1) * num_epochs
    num_warmup_steps = (num_train_steps*0.2) // 100 if args.checkpoint is None else 0
    logging.info('num_data: {}'.format(num_data))
    logging.info('num_train_steps: {}'.format(num_train_steps))
    logging.info('num_warmup_steps: {}'.format(num_warmup_steps))

    optimizer = AdamW(model.parameters(), lr=6.25e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

    start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    experiment_key = 'gpt2_paraphrase'
    experiment_key += '' if not is_toy else '_toy'
    experiment_key += '' if args.checkpoint is None else '_continued'
    experiment_key += '_{}'.format(start_datetime)
    save_dir = './checkpoints/{}/'.format(experiment_key)
    os.makedirs(save_dir)

    tb_writer = SummaryWriter('runs/{}'.format(experiment_key))

    logging.info("Start training")
    total_step = 0
    for ep in range(1, num_epochs+1):
        step = 0
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        for begin_loc in tqdm(range(0, len(train_dataset), batch_size)):
            total_step += 1
            step += 1
            end_loc = begin_loc + batch_size
            samples = train_dataset[indices[begin_loc:end_loc]]
            outputs = model(**samples)
            #loss, logits = outputs[:2]
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()

            tb_writer.add_scalar('Loss/train', loss.item(), total_step)
            if step % 20000 == 0:
                logging.info("{} step of {} epoch".format(step, ep))
                ckp = save_dir + 'ep{}_{}k_steps'.format(ep, step // 1000)
                os.makedirs(ckp)
                model.save_pretrained(ckp)
                tokenizer.save_pretrained(ckp)
                logging.info("Model saved")
                ppl = get_ppl(model, dev_dataset, eval_batch_size)
                logging.info("PPL = {}".format(ppl))
                tb_writer.add_scalar('Loss/dev_log_ppl', np.log(ppl), total_step)

        logging.info("End of {} epoch".format(ep))
        ckp = save_dir + 'ep{}'.format(ep)
        os.makedirs(ckp)
        model.save_pretrained(ckp)
        tokenizer.save_pretrained(ckp)
        logging.info("Model saved")
        ppl = get_ppl(model, dev_dataset, eval_batch_size)
        logging.info("PPL = {}".format(ppl))
        tb_writer.add_scalar('Loss/dev_log_ppl', np.log(ppl), total_step)
    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-medium',
                        help='pretrained model name (only gpt available)')
    parser.add_argument('--train_data', type=str,
                        default='./data/train_only/train_preparation.txt',
                        help='train dataset file')
    parser.add_argument('--dev_data', type=str,
                        default='./data/train_only/dev_preparation.txt',
                        help='dev dataset file')
    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stderr)
    logging.getLogger("UnsupervisedParaphraseGeneration_TRAIN.*").setLevel(log_level)

    ### Reproducability
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.backends.cudnn.deterministic = True  # It may have a negative performance impact
    #torch.backends.cudnn.benchmark = False

    train(args, args.model, args.train_data, args.dev_data, args.toy)

