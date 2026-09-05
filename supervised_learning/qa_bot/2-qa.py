#!/usr/bin/env python3
"""
Module that defines a function to run an interactive loop that
answers questions about a reference text using a pre-trained BERT
question-answering model.
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference text in an interactive loop.

    Prompts the user with 'Q:' and looks up the answer to their
    question within the given reference text using the
    question_answer function. Prints the answer prefixed with 'A:'.
    If no answer can be found, prints a message indicating the
    question was not understood. The loop exits when the user
    types 'exit', 'quit', 'goodbye', or 'bye' (case insensitive),
    printing 'A: Goodbye'.

    Args:
        reference (str): the reference text to search for answers.

    Returns:
        None
    """
    exit_words = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input('Q: ')

        if question.strip().lower() in exit_words:
            print('A: Goodbye')
            break

        answer = question_answer(question, reference)

        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: {}'.format(answer))
