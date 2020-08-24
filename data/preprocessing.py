import argparse
import csv
import random

from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import GPT2Tokenizer

from eda import synonym_replacement

english_stopwords = stopwords.words('english')

# Stopwords from case study of the paper
# 1. From case study
english_stopwords += ['someone', 'something', 'make', 'see']
# 2. From possible candidates
english_stopwords += ['everything']
# 3. Similar words from those of case study
english_stopwords += ['anyone', 'anything', 'everyone']

tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()


def remove_stopwords(sentence):
    sentence = tokenizer.tokenize(sentence)
    sentence = [word for word in sentence
                if word.lower() not in english_stopwords]
    sentence = ' '.join(sentence)
    sentence = sentence.replace("''", '"').replace('``', '"')
    sentence = detokenizer.detokenize(sentence.split())
    return sentence


def sentence_noising(sentence, shuffle_ratio=0.2, replace_ratio=0.2):
    # 1. Synonym replacement
    words = sentence.split()
    n_sr = max(1, int(len(words)*shuffle_ratio))
    words = synonym_replacement(words, n_sr)

    # 2. Random shuffling
    if random.random() < shuffle_ratio:
        random.shuffle(words)

    return ' '.join(words)


def data_preparation(args):
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    data = []
    with open(args.input) as f:
        skipped = 0
        for line in f:
            sentence = line.strip()
            corrupted_sentence = remove_stopwords(sentence)
            write_line = corrupted_sentence + '\n' + sentence
            if len(gpt_tokenizer.encode(write_line)) < args.max_length:
                data.append([corrupted_sentence, sentence])
            else:
                skipped += 1
    print("Skipped: {}".format(skipped))

    with open(args.output, 'w') as wf:
        writer = csv.writer(wf)
        for corrupted, sentence in data:
            writer.writerow([corrupted, sentence])

    for i in range(args.num_generate):
        filename = args.corrupted_output.format(i)
        with open(filename, 'w') as wf:
            writer = csv.writer(wf)
            for corrupted, sentence in data:
                corrupted = sentence_noising(corrupted)
                writer.writerow([corrupted, sentence])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='input file')
    parser.add_argument('--output', type=str, required=True,
                         help='output sentence after removing stop words')
    parser.add_argument('--corrupted_output', type=str, default=None,
                         help='output sentences after all corruptions')

    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_generate', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    random.seed(args.seed)

    if args.corrupted_output is None:
        args.corrupted_output = args.output + '.{}'

    data_preparation(args)
