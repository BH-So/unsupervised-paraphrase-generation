import argparse
import csv
import json
import os
from datetime import datetime
import random
import logging

import numpy as np
import torch
from model.gpt2_trainer import FinetuneGPT2


start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def inference(args):
    gpt_model = FinetuneGPT2(args)
    gpt_model.build_model(args.checkpoint, with_tokenizer=False)

    sentences = []
    with open(args.data_path) as f:
        reader = csv.reader(f)
        for corrupted, _ in reader:
            sentences.append(corrupted)

    if args.toy is True:
        sentences = sentences[:4]

    logging.info("START INFERENCE")
    seq_list = gpt_model.generate_text(
        sentences,
        max_length=args.max_length,
        decoding=args.decoding,
        suffix='[SEP]'
    )
    logging.info("DONE INFERENCE")
    logging.info("Save to {}".format(args.save))
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, 'w') as f:
        for idx, generated in enumerate(seq_list):
            if isinstance(generated, list):
                for seq in generated:
                    f.write('{}\t{}\n'.format(idx, seq))
            else:
                f.write('{}\t{}\n'.format(idx, generated))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./data/QQP_split/test_input_preprocessed.txt',
                        help='Dataset file to paraphrase')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to LOAD model checkpoint')
    parser.add_argument('--model', type=str, default='gpt2-medium',
                        help='pretrained model name (to load tokenizer)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=str, default=None,
                        help='File name to save generated sentences')
    parser.add_argument('--log', type=str, default=None,
                        help='Log filename')

    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum number of tokens for each sequence')

    parser.add_argument('--decoding', type=str, default='sampling',
                        help='{greedy, sampling, beam}')
    parser.add_argument('--beam_size', type=int, default=8,
                        help='Beam size for beam search decoding')
    parser.add_argument('--k', type=int, default=0,
                        help='k for top-k sampling (0 for deactivate)')
    parser.add_argument('--p', type=float, default=1.0,
                        help='p for necleus (top-p) sampling')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for sampling-based decoding')
    parser.add_argument('--num_generate', type=int, default=1,
                        help='How many sequences are generated')

    parser.add_argument('--tag', type=str, default='',
                        help='Add a suffix of checkpoints')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    args.decoding_name = args.decoding
    if args.decoding == 'beam':
        args.decoding_name += '-{}'.format(args.beam_size)
        raise NotImplementedError  # TODO
    elif args.decoding == 'sampling':
        args.decoding_name = 'top-{}'.format(args.k)
        args.decoding_name += '-p{}'.format(args.p).replace('.', '_')
        args.decoding_name += '-T{}'.format(args.temperature).replace('.', '_')

    filename = "inferenced_{}_seed{}_{}".format(
            args.decoding_name, args.seed, args.tag + '_' + start_datetime)

    if args.save is None:
        args.save = "./results/{}.txt".format(filename)
    if args.log is None:
        args.log = './logs/{}.log'.format(filename)

    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format=log_format, filename=args.log)
    logging.getLogger().setLevel(log_level)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info('Parsed args: ' + json.dumps(dict(args.__dict__), indent=2))

    inference(args)
