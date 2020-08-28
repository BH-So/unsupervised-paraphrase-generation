import csv
import logging

# import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class QQPDataset(Dataset):
    def __init__(self, tokenizer, filename,
                 max_length=512, device='cuda',
                 is_inference=False, load_augmented=False, is_toy=False):
        self.tokenizer = tokenizer
        self.filename = filename
        self.max_length = max_length
        self.device = device
        self.is_inference = bool(is_inference)
        self.is_toy = is_toy

        self.epoch = 1 if load_augmented is True else None
        self.load_dataset(epoch=self.epoch)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_ids = self.input_ids[idx, :].to(self.device)
        samples = {
            'input_ids': input_ids,
        }
        if self.is_inference is False:
            samples['attention_mask'] = \
                    self.attention_mask[idx, :]
            samples['labels'] = self.labels[idx, :]
        return samples

    def load_dataset(self, epoch=None):
        filename = self.filename
        if epoch is not None:
            filename += '.{}'.format(epoch-1)
        logging.info("Loading data from {}".format(filename))

        data = []
        with open(filename) as f:
            reader = csv.reader(f)
            for corrupted, sentence in reader:
                data.append([corrupted, sentence])
                if self.is_toy is True:
                    break

        tokens_list, labels_list = [], []
        for corrupted, sentence in data:
            tokens, labels = self.formatting(corrupted, sentence)
            tokens_list.append(tokens)
            labels_list.append(labels)
        sentences = [self.tokenizer.decode(tokens)
                     for tokens in tokens_list]
        encodings = self.tokenizer(
            sentences, return_tensors='pt', truncation=True,
            padding='max_length', max_length=self.max_length)

        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

        if self.is_inference is False:
            self.labels = torch.tensor(labels_list, dtype=torch.long)

    def formatting(self, input_text, target_text):
        input_tokens = self.tokenizer.encode(input_text)
        target_tokens = self.tokenizer.encode(target_text)

        tokens = [self.tokenizer.bos_token_id] + input_tokens \
            + [self.tokenizer.sep_token_id] + target_tokens \
            + [self.tokenizer.eos_token_id]

        labels = [-100] * (len(input_tokens) + 2) \
            + target_tokens + [self.tokenizer.eos_token_id] \
            + [-100] * (self.max_length - len(tokens))
        labels = labels[:self.max_length]
        return tokens, labels
