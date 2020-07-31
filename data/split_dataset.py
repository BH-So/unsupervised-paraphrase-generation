import csv
import random

# Read file
paraphrase_labeled_file = 'QQP/train.csv'
paraphrase_unlabeled_file = 'QQP/test.csv'
# Write files
train_file = 'train.txt'
dev_file = 'dev.txt'
test_file = 'test.txt'

test_pair_num = 30000
dev_num = 60000


if __name__ == '__main__':
    with open(paraphrase_labeled_file) as f:
        reader = csv.reader(f)
        questions = []
        paraphrases = []
        header = next(reader)
        for row in reader:
            _, _, _, question1, question2, is_duplicate = row
            questions.append(question1)
            questions.append(question2)
            if is_duplicate == '1':
                paraphrases.append([question1, question2])

    with open(paraphrase_unlabeled_file) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            _, question1, question2 = row
            questions.append(question1)
            questions.append(question2)

    questions = list(set(questions))
    print("# questions: {}".format(len(questions)))
    print("# paraphrases: {}".format(len(paraphrases)))

    test_questions = random.sample(paraphrases, test_pair_num)
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for q1, q2 in test_questions:
            writer.writerow([q1, q2])
            for q in [q1, q2]:
                try:
                    questions.remove(q)
                except ValueError:
                    pass

    random.shuffle(questions)
    with open(dev_file, 'w') as f:
        for question in questions[:dev_num]:
            f.write(question + '\n')
    with open(train_file, 'w') as f:
        for question in questions[dev_num:]:
            f.write(question + '\n')

