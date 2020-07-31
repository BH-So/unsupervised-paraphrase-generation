import argparse
import datetime
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
from load_dataset import HuggingfaceDataset

import logging
logging.basicConfig(level=logging.INFO)

device = 'cuda'


def get_ppl(model, dataset, batch_size=32):
    _ = model.eval()
    with torch.no_grad():
        lls = []
        for begin_loc in range(0, len(dataset), batch_size):
            end_loc = begin_loc + batch_size
            input_ids, attention_mask, labels = dataset[begin_loc:end_loc]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            log_likelihood = outputs[0] * input_ids.size(0)
            lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / len(dataset))
    _ = model.train()
    return ppl.item()


def main(model_name, train_filename, dev_filename, is_toy=False):
    num_epochs = 5
    batch_size = 4 # DONE: 4, 2  # FAIL: 8 (with truncate 256)
    eval_batch_size = 4 #8 # FAIL: 16 (with truncate 256)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    special_tokens_dict = {'sep_token': '[SEP]'}
    tokenizer.pad_token = tokenizer.eos_token
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    _ = model.train()

    if is_toy:
        train_filename == dev_filename
    train_dataset = HuggingfaceDataset(tokenizer, train_filename,
                                       device=device, is_toy=is_toy)
    dev_dataset = HuggingfaceDataset(tokenizer, dev_filename,
                                     device=device, is_toy=is_toy)
    num_data = len(train_dataset)
    num_train_steps = ((num_data - 1) // batch_size + 1) * num_epochs
    num_warmup_steps = num_train_steps // 500
    print('num_data: {}'.format(num_data))
    print('num_train_steps: {}'.format(num_train_steps))
    print('num_warmup_steps: {}'.format(num_warmup_steps))

    optimizer = AdamW(model.parameters(), lr=6.25e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

    #experiment_key = 'gpt2_paraphrase'
    experiment_key = 'gpt2_paraphrase_b{}'.format(batch_size)
    experiment_key += '' if not is_toy else '_toy'
    save_dir = './model/{}/'.format(experiment_key)
    os.makedirs(save_dir, exist_ok=True)

    tb_writer = SummaryWriter('runs/{}'.format(experiment_key))


    print("Start training")
    start_datetime = "{}".format(datetime.datetime.now()).replace(' ', '_')
    n_iter = 0
    for ep in range(1, num_epochs+1):
        n_iter_ep = 0
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        for begin_loc in tqdm(range(0, len(train_dataset), batch_size)):
            n_iter += 1
            n_iter_ep += 1
            end_loc = begin_loc + batch_size
            input_ids, attention_mask, labels = train_dataset[indices[begin_loc:end_loc]]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            #loss, logits = outputs[:2]
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()

            tb_writer.add_scalar('Loss/train', loss.item(), n_iter)
            if n_iter_ep % 20000 == 0:
                print("{} step of {} epoch".format(n_iter_ep, ep))
                torch.save(model.state_dict(),
                           save_dir + '{}ep_{}steps_{}'.format(ep, n_iter_ep, start_datetime))
                print("Model saved")
                ppl = get_ppl(model, dev_dataset, eval_batch_size)
                print("PPL = {}".format(ppl))
                tb_writer.add_scalar('Loss/dev_log_ppl', np.log(ppl), n_iter)

        print("End of {} epoch".format(ep))
        torch.save(model.state_dict(), save_dir + '{}ep_{}'.format(ep, start_datetime))
        print("Model saved")
        ppl = get_ppl(model, dev_dataset, eval_batch_size)
        print("PPL = {}".format(ppl))
        tb_writer.add_scalar('Loss/dev_log_ppl', np.log(ppl), n_iter)
    tb_writer.close()
    #training_args = TrainingArguments(
    #    output_dir='./results',          # output directory
    #    num_train_epochs=5,              # total # of training epochs
    #    per_device_train_batch_size=16,  # batch size per device during training
    #    per_device_eval_batch_size=64,   # batch size for evaluation
    #    warmup_steps=2000,               # number of warmup steps for learning rate scheduler
    #    weight_decay=0.01,               # strength of weight decay
    #    logging_dir='./logs',            # directory for storing logs
    #)
    #### Experimental setup of OpenAI GPT fine-tuning ###
    ## learning rate = 6.25e-5
    ##       (2.5e-4 in training)
    ## weight decay = lienar lr decay schedule with warmup over 0.2% of training
    ##       -> lambda = 0.5
    ##       (2000 warmup steps in training)
    ##       (annealed to 0 using a cosine schedule in training)
    ## stride = 512
    ## batch size = 32
    ##       (64 in training)
    ## L2 norm = 0.1 (modified version from "Fixing weight decay regularization in adam")
    ##       ->  w = 0.01 on all non bias or gain wiehgts
    ## dropout = 0.1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-medium',
                        help='pretrained model name (only gpt available)')
    parser.add_argument('--train_data', type=str,
                        default='data/train_only/train_preparation.txt',
                        help='input file')
    parser.add_argument('--dev_data', type=str,
                        default='data/train_only/dev_preparation.txt',
                        help='dev file')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    ### Reproducability
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.backends.cudnn.deterministic = True  # It may have a negative performance impact
    #torch.backends.cudnn.benchmark = False

    main(args.model, args.train_data, args.dev_data, args.toy)
