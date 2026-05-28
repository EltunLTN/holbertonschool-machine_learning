#!/usr/bin/env python3
"""Module to plot a stacked bar graph of fruit per person."""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot a stacked bar graph of fruit quantities per person."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    labels = ['apples', 'bananas', 'oranges', 'peaches']
    x = np.arange(len(people))
    bottom = np.zeros(3)

    for i in range(4):
        plt.bar(x, fruit[i], width=0.5, bottom=bottom,
                color=colors[i], label=labels[i])
        bottom += fruit[i]

    plt.xticks(x, people)
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()
