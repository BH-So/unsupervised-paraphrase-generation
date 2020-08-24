
### Download dataset

Download QQP (Quora Question Pair) dataset from [kaggle] (http://www.kaggle.com/c/quora-question-pairs)


### Data preprocessing

1. Split dataset: split\_dataset.py
  - For small dataset, do not use test.csv of QQP

2. Change sentence to sentence reconstruction task format: preprocessing.py
```
python preprocessing.py --input ${input file} --output ${output file}
```

### Code citation

We are referring to [here] (https://github.com/jasonwei20/eda_nlp/) for `synonym_replacement` function

