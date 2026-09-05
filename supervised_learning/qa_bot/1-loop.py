#!/usr/bin/env python3
"""
Module that implements a simple interactive loop that prompts the
user with 'Q:' and responds with 'A:', exiting when the user types
one of a set of exit keywords.
"""


if __name__ == '__main__':
    exit_words = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input('Q: ')

        if question.strip().lower() in exit_words:
            print('A: Goodbye')
            break

        print('A:')
