import argparse
import csv
import json
import os
from datetime import datetime
import random
import logging
from collections import defaultdict

import nltk
from nltk.translate.meteor_score import single_meteor_score as meteor
import numpy as np
import torch
from fast_bleu import SelfBLEU
from rouge_score import rouge_scorer


available_metrics = ('self-bleu', 'meteor', 'rouge')
start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def evaluate(args):
    metrics = args.metrics.lower().split(',')

    sentences = defaultdict(list)
    with open(args.generated) as f:
        lines = [line.strip().split('\t') for line in f]
        lines = [(int(row[0]), row[1]) for row in lines]
        for idx, sent in lines:
            sentences[idx].append(sent)
    logging.debug("Example generated sentences: {}".format(sentences[0]))
    logging.debug("Read {} generated sentences".format(len(sentences)))

    gt_sentences = []
    if args.ground_truth is not None:
        with open(args.ground_truth) as f:
            reader = csv.reader(f)
            for _, sent in reader:
                gt_sentences.append(sent)
        logging.debug("Example gt sentences: {}".format(gt_sentences[0]))
        logging.debug("Read {} gt sentences".format(len(gt_sentences)))

    if args.toy is True:
        gt_sentences = gt_sentences[:4]
        sentences = {i: sentences[i] for i in range(4)}

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, 'w') as f:
        f.write("Generated sentences file: {}\n".format(args.generated))
        if args.ground_truth is not None:
            f.write("Ground truth file: {}\n".format(args.ground_truth))
        for metric in metrics:
            logging.debug("START EVALUATION: {}".format(metric))

            cnt = len(sentences)
            if metric == 'meteor':
                # Calculate METEOR score for each paraphrases
                meteor_scores = defaultdict(list)
                for idx, candidates in sentences.items():
                    gt = gt_sentences[idx]
                    for cand in candidates:
                        score = meteor(gt, cand)
                        meteor_scores[idx].append((score, cand))
                logging.debug("Example METEOR scores: {}".format(meteor_scores[0]))

                # Get the best METEOR score for each input
                for key in meteor_scores.keys():
                    meteor_scores[key].sort(key=lambda row: -row[0])
                best_score = sum([score_list[0][0] for score_list in meteor_scores.values()]) / cnt
                logging.info("Best METEOR:  {}".format(best_score))

                # Get top 3 METEOR scores for each input
                top3_score = sum([sum([score for score, _ in row[:3]]) / len(row[:3])
                                  for row in meteor_scores.values()]) / cnt
                logging.debug("Example top 3 METEOR scores: {}".format(meteor_scores[0][:3]))
                logging.info("Top 3 METEOR: {}".format(top3_score))

                # Self-BLEU among top 3 paraphrases
                sbleu = 0
                weights = {'4gram': (0.25, 0.25, 0.25, 0.25)}
                for val in meteor_scores.values():
                    refs = [sent for _, sent in val[:3]]
                    calculator = SelfBLEU(refs, weights=weights)
                    score_list = calculator.get_score()['4gram']
                    sbleu += sum(score_list) / len(score_list)
                logging.info("self-BLEU among top 3: {}".format(sbleu / cnt))
            elif metric == 'rouge':
                # Calculate ROUGE score for each paraphrases
                rouge1_scores = defaultdict(list)
                rouge2_scores = defaultdict(list)
                rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
                for idx, candidates in sentences.items():
                    gt = gt_sentences[idx]
                    for cand in candidates:
                        scores = rouge.score(gt, cand)
                        rouge1_scores[idx].append((scores['rouge1'].fmeasure, cand))
                        rouge2_scores[idx].append((scores['rouge2'].fmeasure, cand))
                logging.debug("Example ROUGE-1 scores: {}".format(rouge1_scores[0]))
                logging.debug("Example ROUGE-2 scores: {}".format(rouge2_scores[0]))

                # Get the best ROUGE score for each input
                for key in rouge1_scores.keys():
                    rouge1_scores[key].sort(key=lambda row: -row[0])
                for key in rouge2_scores.keys():
                    rouge2_scores[key].sort(key=lambda row: -row[0])
                best_rouge1 = sum([score_list[0][0] for score_list in rouge1_scores.values()]) / cnt
                best_rouge2 = sum([score_list[0][0] for score_list in rouge2_scores.values()]) / cnt
                logging.info("Best ROUGE-1: {}".format(best_rouge1))
                logging.info("Best ROUGE-2: {}".format(best_rouge2))

                # Get top 3 ROUGE scores for each input
                top3_rouge1 = sum([sum([score for score, _ in row[:3]]) / len(row[:3])
                                  for row in rouge1_scores.values()]) / cnt
                top3_rouge2 = sum([sum([score for score, _ in row[:3]]) / len(row[:3])
                                  for row in rouge2_scores.values()]) / cnt
                logging.debug("Example top 3 ROUGE-1 scores: {}".format(rouge1_scores[0][:3]))
                logging.debug("Example top 3 ROUGE-2 scores: {}".format(rouge2_scores[0][:3]))
                logging.info("Top 3 ROUGE-1: {}".format(top3_rouge1))
                logging.info("Top 3 ROUGE-2: {}".format(top3_rouge2))
            else:
                logging.warning("Not available: {}".format(metric))
                continue
            logging.debug("DONE EVALUATION: {}".format(metric))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=str,
                        default='./results/filtered/inferenced.txt',
                        help='Generated sentences (paraphrases)')
    parser.add_argument('--ground_truth', type=str, default=None,
                        help='Ground turth paraphrases')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=str, default=None,
                        help='File name to save generated sentences')
    parser.add_argument('--log', type=str, default=None,
                        help='Log filename')

    parser.add_argument('--metrics', type=str, default='meteor',
                        help='[{}]'.format(', '.join(available_metrics)))
    parser.add_argument('--self-bleu-samples', type=int, default=None,
                        help='Number of samples to measure self-bleu')

    parser.add_argument('--tag', type=str, default='',
                        help='Add a suffix of checkpoints')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()


    filename = args.generated.split('/')[-1].split('.')[0]
    filename = "evaluation_results_{}_{}".format(
            filename, args.tag + '_' + start_datetime)

    if args.save is None:
        args.save = "./results/evaluation/{}.txt".format(filename)
    if args.log is None:
        args.log = './logs/{}.log'.format(filename)

    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format=log_format, filename=args.log)
    logging.getLogger().setLevel(log_level)
    if args.verbose is True:
        stdout_handler = logging.StreamHandler(sys.stdout)
        logging.getLogger().addHandler(stdout_handler)

    ### Reproducibility
    random.seed(args.seed)

    logging.info('Parsed args: ' + json.dumps(dict(args.__dict__), indent=2))

    evaluate(args)
