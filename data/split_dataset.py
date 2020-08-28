import csv
import os
import random

# Read file
paraphrase_labeled_file = 'QQP/train.csv'
paraphrase_unlabeled_file = 'QQP/test.csv'
# Write files
train_file = 'QQP_split/train.txt'
dev_file = 'QQP_split/dev.txt'
test_input_file = 'QQP_split/test_input.txt'
test_target_file = 'QQP_split/test_target.txt'

test_pair_num = 30000
dev_num = 60000
unlabeled_used = 300000

random.seed(1234)
os.mkdir("QQP_split")


def data_cleansing(text):
    text = ' '.join(text.split())
    return text


if __name__ == '__main__':
    with open(paraphrase_labeled_file) as f:
        reader = csv.reader(f)
        questions_1 = []
        questions_2 = []
        paraphrases = []
        header = next(reader)
        for idx, row in enumerate(reader):
            _, _, _, question1, question2, is_duplicate = row
            questions_1.append(data_cleansing(question1))
            questions_2.append(data_cleansing(question2))
            if is_duplicate == '1':
                paraphrases.append(idx)

    with open(paraphrase_unlabeled_file) as f:
        reader = csv.reader(f)
        header = next(reader)
        unlabeled_questions = []
        for idx, row in enumerate(reader):
            _, question1, _ = row
            unlabeled_questions.append(data_cleansing(question1))
            if idx >= unlabeled_used:
                break

    test_indices = random.sample(paraphrases, test_pair_num)
    test_questions = [questions_1[idx] for idx in test_indices] \
        + [questions_2[idx] for idx in test_indices]
    test_questions = set(test_questions)

    questions = set(questions_1 + unlabeled_questions)
    training_questions = list(questions - set(test_questions))

    print("# questions: {}".format(len(questions)))
    print("# training questions: {}".format(len(training_questions)))

    with open(test_input_file, 'w', newline='') as f_i, \
            open(test_target_file, 'w', newline='') as f_t:
        for idx in test_indices:
            q1 = questions_1[idx]
            q2 = questions_2[idx]
            f_i.write(q1 + '\n')
            f_t.write(q2 + '\n')

    # Converting set to list could be shuffled the order,
    # but we want the same result with the same random seed
    training_questions.sort()

    random.shuffle(training_questions)
    with open(dev_file, 'w') as f:
        for question in training_questions[:dev_num]:
            f.write(question + '\n')
    with open(train_file, 'w') as f:
        for question in training_questions[dev_num:]:
            f.write(question + '\n')
