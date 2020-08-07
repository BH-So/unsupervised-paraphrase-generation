import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class QQPDataset(Dataset):
    def __init__(self, tokenizer, filename,
                 max_length=256, device='cuda', is_toy=False):
        self.max_length = max_length
        encodings = self.load_dataset(tokenizer, filename, is_toy)
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.device = device

        sep_token_id = tokenizer.encode(tokenizer.sep_token)[0]
        self.labels = self.compute_labels(
            self.input_ids, self.attention_mask, sep_token_id)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_ids = self.input_ids[idx, :].to(self.device)
        attention_mask = self.attention_mask[idx, :].to(self.device)
        labels = self.labels[idx, :].to(self.device)
        samples = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        return samples

    def load_dataset(self, tokenizer, filename, is_toy=False):
        with open(filename) as f:
            lines = [line.strip() for line in f]
        if is_toy:
            # lines = lines[:64]
            lines = lines[:1]
        tokens = [tokenizer.encode(line) + [tokenizer.eos_token_id]
                  for line in lines]
        sentences = [tokenizer.decode(x) for x in tokens]
        encodings = tokenizer(sentences, return_tensors='pt', truncation=True,
                              padding='max_length', max_length=self.max_length)
        return encodings

    def compute_labels(self, input_ids, attention_mask, sep_token_id):
        labels = input_ids.clone()
        sep_token_exist, sep_tok_positions = np.where(labels == sep_token_id)
        assert(len(sep_token_exist) == len(labels))
        sep_tok_positions = torch.tensor(sep_tok_positions).unsqueeze(1)
        labels[torch.arange(labels.size(1)) <= sep_tok_positions] = -100
        labels[attention_mask == 0] = -100
        return labels
