# QA Bot

This project builds a Question-Answering system on top of a
pre-trained BERT model. It uses the `bert-uncased-tf2-qa` model from
`tensorflow-hub`, tokenized with the `bert-large-uncased-whole-word-
masking-finetuned-squad` `BertTokenizer` from `huggingface/transformers`,
to find the snippet of text within a reference document that best
answers a given question.

## Requirements

* Ubuntu 20.04 LTS
* python3 (3.9)
* numpy 1.25.2
* tensorflow 2.15
* tensorflow-hub 0.15.0
* transformers 4.44.2
* pycodestyle 2.11.1

### Installing dependencies

```
pip install --user tensorflow-hub==0.15.0
pip install --user transformers==4.44.2
```

### Data

This project uses a collection of Holberton USA Zendesk Articles
(`ZendeskArticles.zip`), which should be unzipped into a
`ZendeskArticles` directory at the root of this project.

## Tasks

### 0. Question Answering

`0-qa.py` contains the function `question_answer(question, reference)`
that finds a snippet of text within a reference document to answer a
question:

* `question` is a string containing the question to answer
* `reference` is a string containing the reference document from
  which to find the answer
* Returns a string containing the answer, or `None` if no answer is
  found

Example:

```
$ cat 0-main.py
#!/usr/bin/env python3

question_answer = __import__('0-qa').question_answer

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

print(question_answer('When are PLDs?', reference))
$ ./0-main.py
on - site days from 9 : 00 am to 3 : 00 pm
$
```