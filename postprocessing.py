import argparse
import os
import json
from datetime import datetime
import logging
from collections import defaultdict

import torch
import torch.multiprocessing
from utils.edit_distance import levenshtein
from sentence_transformers import SentenceTransformer


torch.multiprocessing.set_sharing_strategy('file_system')

# Sentence-BERT generate a lot of INFO logs
# Make these logs to silence
logging.INFO_ = logging.INFO + 5
logging.addLevelName(logging.INFO_, 'INFO_')
def _info25(msg, *args, **kwargs):
    logging.log(logging.INFO_, msg, *args, **kwargs)
logging.info_ = _info25

start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_cosine_similarity(embedding1, embedding2):
    cos = torch.nn.CosineSimilarity()
    t1 = torch.tensor(embedding1).unsqueeze(0)
    t2 = torch.tensor(embedding2).unsqueeze(0)
    sim = cos(t1, t2)
    return sim.item()


def filter_special_tokens(args, sent):
    while sent.endswith(args.eos):
        sent = sent[:-len(args.eos)].strip()
    return sent


def candidate_filtering(args):
    max_length = args.max_length
    sep_token = args.sep_token
    min_diff = args.min_diff
    threshold = args.threshold

    with open(args.input) as f:
        inputs = [line.strip() for line in f]
    logging.info_("Read {} lines of model input: {}".format(
        len(inputs), args.input))

    with open(args.paraphrase) as f:
        paraphrase_lines = defaultdict(set)
        while True:
            line = f.readline()
            if not line:
                break
            try:
                idx, sent = line.strip().split('\t')
            except ValueError:
                continue
            sent = sent.split(sep_token)[1].strip()
            sent = filter_special_tokens(args, sent)
            paraphrase_lines[int(idx)].add(sent)
    logging.info_("Read model outputs from {} inputs: {}".format(
            len(paraphrase_lines), args.paraphrase))

    if args.toy is True:
        inputs = inputs[:8]
        paraphrase_lines = paraphrase_lines[:8]

    embedder = SentenceTransformer(args.model)
    input_embeddings = embedder.encode(inputs)
    with open(args.output, 'w') as wf:
        indices = list(paraphrase_lines.keys())
        indices.sort()
        cnt = 0
        for idx in indices:
            scores = []
            for paraphrase in paraphrase_lines[idx]:
                cnt += 1
                logging.info_("Count={}, Index={}:\n\tInput:\t{}\n\tGen:\t{}".format(
                        cnt, idx, inputs[idx], paraphrase))

                # Cosine similarity of embeddings using Sentence-BERT
                embedding = embedder.encode(paraphrase)[0]
                sim = get_cosine_similarity(embedding, input_embeddings[idx])
                logging.info_("Sentence-BERT CosSim: {}".format(sim))

                # Character-level Levenshtein distance
                distance = levenshtein(paraphrase, inputs[idx])
                logging.info_("Character diffrence: {}".format(distance))
                # scores.append([sim, distance, paraphrase])

                # Filter if embedding is not similar enough
                #   or Levenshtein distance is too small
                if sim < threshold:
                    logging.info_(" (Filtered) Different meaning")
                    continue
                scores.append([sim, distance, paraphrase])
                if distance < min_diff:
                    logging.info_(" (Filtered) Too similar to input")
                    continue

                # Write the paraphrase not filtered
                if args.best_only is False:
                    wf.write('{}\t{}\n'.format(idx, paraphrase))
            if args.best_only is True:
                if len(scores) == 0:
                    logging.warning("There is no paraphrase which is similar enough")
                    continue
                scores = [[100*(dist>=min_diff)+sim, text]
                          for sim, dist, text in scores]
                scores.sort(key=lambda row: row[0], reverse=True)
                best_paraphrase = scores[0][1]
                logging.info_("Write the best: {}".format(best_paraphrase))
                wf.write('{}\t{}\n'.format(idx, best_paraphrase))
    logging.info_("Done postprocessing!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='./data/QQP_split/test_input.txt',
                        help='Original sentence file')
    parser.add_argument('--paraphrase', type=str, required=True,
                        help='Paraphrased sentence file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to write')
    parser.add_argument('--log', type=str, default=None,
                        help='Log filename')

    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Cosine similarity threshold (Sentence-BERT)')
    parser.add_argument('--min_diff', type=int, default=6,
                        help='Minimum (character-level) Levenshtein distance')
    parser.add_argument('--best_only', action='store_true',
                        help='Remain the best cosine similarity (>= threshold)')

    parser.add_argument('--model', type=str,
                        default='bert-base-nli-stsb-mean-tokens',
                        help='Sentence-BERT model name')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum number of tokens for Sentence-BERT')
    parser.add_argument('--sep_token', type=str, default='[SEP]')
    parser.add_argument('--eos', type=str, default='<|endoftext|>',
                        help='EOS token to ignore if exist in paraphrases')

    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()


    filename = os.path.basename(args.paraphrase).split('.')[0]
    filename = 'postprocess_{}'.format(filename)
    filename += '_MinDiff{}_thres{}_{}_{}'.format(
            args.min_diff, args.threshold, args.tag, start_datetime)

    if args.output is None:
        args.output = './results/filtered/{}.txt'.format(filename)

    if args.log is None:
        args.log = './logs/{}.log'.format(filename)
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.DEBUG if args.debug else logging.INFO_
    log_level_utils = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format=log_format, filename=args.log)
    logging.getLogger().setLevel(log_level)
    util_loggers = [
        "modeling_utils",
        "transformers.configuration_utils",
        "transformers.tokenization_utils_base",
    ]
    for name in util_loggers:
        logging.getLogger(name).setLevel(log_level_utils)

    logging.info_('Parsed args: ' + json.dumps(dict(args.__dict__), indent=2))

    candidate_filtering(args)
