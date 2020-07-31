import argparse
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from eda import synonym_replacement

sep_token = '[SEP]'
english_stopwords = stopwords.words('english')


def remove_stopwords(sentence):
    sentence = [word for word in sentence if word not in english_stopwords]
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
    sentence = word_tokenize(sentence)

    ### 1. Remove stop words
    sentence = remove_stopwords(sentence)

    ### 2. Randomly shuffle
    n = int(round(len(sentence) * shuffle_ratio, 0))
    if n >= 2:
        sentence = shuffle_words(sentence, n)

    ### 3. Randomly replace to synonyms
    n = int(round(len(sentence) * replace_ratio, 0))
    if n >= 1:
        sentence = synonym_replacement(sentence, n)

    sentence = ' '.join(sentence)
    return sentence


def data_preparation(inputfile, outputfile):
    with open(inputfile) as f, open(outputfile, 'w') as wf:
        for line in f:
            sentence = line.strip()
            noised_sentence = sentence_noising(sentence)
            wf.write(noised_sentence + sep_token + sentence + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='input file')
    parser.add_argument('--output', type=str, default=None, help='output file')
    args = parser.parse_args()
    data_preparation(args.input, args.output)
