import argparse
import random

from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import GPT2Tokenizer

from eda import synonym_replacement

english_stopwords = stopwords.words('english')

### Stopwords from case study of the paper
# From case study
english_stopwords += ['someone', 'something', 'make', 'see']
# From possible candidates
english_stopwords += ['everything']
# Similar words from those of case study
english_stopwords += ['anyone', 'anything', 'everyone']

tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()


def remove_stopwords(sentence):
    sentence = [word for word in sentence
                if word.lower() not in english_stopwords]
    return sentence


def shuffle_words(sentence, n):
    shuffled_sentence = sentence.copy()
    replace_indices = random.sample(range(len(sentence)), n)
    mapping = replace_indices.copy()
    random.shuffle(replace_indices)
    mapping = {idx1: idx2 for idx1, idx2 in zip(mapping, replace_indices)}
    for idx1, idx2 in mapping.items():
        shuffled_sentence[idx1] = sentence[idx2]
    return shuffled_sentence


def sentence_noising(sentence, shuffle_ratio=0.2, replace_ratio=0.2):
    sentence = tokenizer.tokenize(sentence)

    # 1. Remove stop words
    sentence = remove_stopwords(sentence)

    # 2. Randomly shuffle
    n = int(round(len(sentence) * shuffle_ratio, 0))
    if n >= 2:
        sentence = shuffle_words(sentence, n)

    # 3. Randomly replace to synonyms
    n = int(round(len(sentence) * replace_ratio, 0))
    if n >= 1:
        sentence = synonym_replacement(sentence, n)

    sentence = ' '.join(sentence)
    sentence = sentence.replace("''", '"').replace('``', '"')
    sentence = detokenizer.detokenize(sentence.split())
    return sentence


def data_preparation(args):
    max_length = args.max_length
    sep_token = args.sep_token
    skip_origin = args.skip_origin

    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    gpt_tokenizer.add_special_tokens({'sep_token': sep_token})
    with open(args.input) as f, open(args.output, 'w') as wf:
        noising_option = {
            'shuffle_ratio': 0 if args.no_shuffle else 0.2,
            'replace_ratio': 0 if args.no_SR else 0.2
        }
        for line in f:
            sentence = line.strip()
            noised_sentence = sentence_noising(sentence, **noising_option)
            write_line = noised_sentence + sep_token
            write_line += ('' if skip_origin else sentence) + '\n'
            if len(gpt_tokenizer.encode(write_line)) <= max_length:
                wf.write(write_line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='input file')
    parser.add_argument('--output', type=str, default=None, help='output file')

    parser.add_argument('--no-SR', action="store_true",
                        help="Whether use synonym replacement or not")
    parser.add_argument('--no-shuffle', action="store_true",
                        help="Whether use word order shuffling or not")

    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--sep_token', type=str, default='[SEP]')
    parser.add_argument('--skip-origin', action="store_true")
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)

    data_preparation(args)
