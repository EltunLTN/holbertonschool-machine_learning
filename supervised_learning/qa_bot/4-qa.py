#!/usr/bin/env python3
"""
Module that defines a function to answer questions from multiple
reference texts, using semantic search to select the most relevant
document and a pre-trained BERT model to extract the answer.
"""
semantic_search = __import__('3-semantic_search').semantic_search
answer_question = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts in an
    interactive loop.

    Prompts the user with 'Q:'. For each question, uses semantic
    search to find the document within corpus_path most relevant
    to the question, then uses a BERT question-answering model to
    extract the answer from that document. Prints the answer
    prefixed with 'A:'. If no answer can be found, prints a message
    indicating the question was not understood. The loop exits when
    the user types 'exit', 'quit', 'goodbye', or 'bye' (case
    insensitive), printing 'A: Goodbye'.

    Args:
        corpus_path (str): the path to the corpus of reference
            documents.

    Returns:
        None
    """
    exit_words = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input('Q: ')

        if question.strip().lower() in exit_words:
            print('A: Goodbye')
            break

        reference = semantic_search(corpus_path, question)
        answer = answer_question(question, reference)

        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: {}'.format(answer))
