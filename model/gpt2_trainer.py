# import argparse
import math
import os
import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
# from transformers import Trainer, TrainingArguments


class FinetuneGPT2(object):
    def __init__(self, args):
        self.args = args
        self.special_tokens_dict = {'sep_token': '[SEP]'}
        self.device = self.args.device
        self.model = self.tokenizer = None
        self.global_step = None

    def build_model(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model)
            self.model = GPT2LMHeadModel.from_pretrained(self.args.model)
            logging.info("Load {} model".format(self.args.model))

            self.tokenizer.add_special_tokens(self.special_tokens_dict)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)
            self.model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
            logging.info("Load model from {}".format(checkpoint_dir))
        self.model.to(self.device)
        self.model.train()

        self.global_step = 0
        self.writer = SummaryWriter(self.args.summary_dir)

    def build_optimizer(self, dataset_size=400000):
        lr = self.args.learning_rate
        self.optimizer = AdamW(self.model.parameters(), lr=lr,
                               weight_decay=0.01)

        num_train_steps = math.ceil(
            dataset_size * self.args.num_epochs / self.args.batch_size)
        self.num_warmup_steps = round(num_train_steps * self.args.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, self.num_warmup_steps, num_train_steps)

        logging.info('num_train_steps : {}'.format(num_train_steps))
        logging.info('num_warmup_steps: {}'.format(self.num_warmup_steps))
        logging.info('Optimizer {}: {}'.format(
            self.optimizer.__class__.__name__, self.optimizer.state_dict()))
        logging.info('Scheduler {}: {}'.format(
            self.scheduler.__class__.__name__, self.scheduler.state_dict()))

    def train(self, samples):
        self.model.train()
        logging.debug(samples)
        outputs = self.model(**samples)
        loss = outputs[0]
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        loss = loss.item()
        self.global_step += 1
        self.writer.add_scalar('Loss/train', loss, self.global_step)

        return loss

    def get_loss(self, dataset, batch_size=None):
        if batch_size is None:
            batch_size = self.args.eval_batch_size
        self.model.eval()
        with torch.no_grad():
            losses = []
            for begin_loc in range(0, len(dataset), batch_size):
                end_loc = begin_loc + batch_size
                samples = dataset[begin_loc:end_loc]
                outputs = self.model(**samples)
                loss = outputs[0] * samples['input_ids'].size(0)
                losses.append(loss)
            avg_loss = torch.stack(losses).sum() / len(dataset)
            ppl = torch.exp(avg_loss).item()

        logging.info("dev PPL = {}".format(ppl))
        self.writer.add_scalar('Loss/dev_ppl', ppl, self.global_step)
        return ppl

    def generate_text(self, input_texts, max_length=1024, decoding='greedy',
                      suffix=''):
        self.model.eval()
        sentences_list = []
        with torch.no_grad():
            kwargs = {'max_length': max_length}
            if decoding == 'sampling':
                kwargs['do_sample'] = True
                kwargs['top_k'] = self.args.k
                kwargs['top_p'] = self.args.p
                kwargs['temperature'] = self.args.temperature
                kwargs['num_return_sequences'] = self.args.num_generate
            for input_text in input_texts:
                sequences = []
                input_text = input_text + suffix
                logging.info('Start to generate from "{}"'.format(input_text))
                input_encoding = self.tokenizer.encode(
                    input_text, return_tensors='pt')
                input_encoding = input_encoding.to(self.device)
                generated_tokens = self.model.generate(
                    input_encoding, **kwargs)
                for tok_seq in generated_tokens:
                    sequence = self.tokenizer.decode(tok_seq)
                    logging.info("Generated text: {}".format(sequence))
                    sequences.append(sequence)
                sentences_list.append(sequences)
        return sentences_list

    def save_model(self, save_dir=None):
        if save_dir is None:
            save_dir = self.args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logging.info("Save model at {}".format(save_dir))