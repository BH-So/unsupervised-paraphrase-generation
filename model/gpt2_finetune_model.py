# import argparse
import math
import os
import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


class FinetuneGPT2(object):
    def __init__(self, args):
        self.args = args
        self.special_tokens_dict = {'sep_token': '[SEP]'}
        self.device = self.args.device
        self.model = self.tokenizer = None
        self.global_step = None

    def build_model(self, checkpoint_dir=None, with_tokenizer=True):
        if checkpoint_dir is None or with_tokenizer is False:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model)
            self.tokenizer.add_special_tokens(self.special_tokens_dict)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)

        if checkpoint_dir is None:
            self.model = GPT2LMHeadModel.from_pretrained(self.args.model)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.info("Load {} model".format(self.args.model))
        else:
            self.model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
            logging.info("Load model from {}".format(checkpoint_dir))
        self.model.to(self.device)
        self.model.train()

        self.global_step = 0
        if hasattr(self.args, 'summary_dir'):
            self.writer = SummaryWriter(self.args.summary_dir)

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
