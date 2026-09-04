#!/usr/bin/env python3
"""
Module for the epsilon-greedy action selection strategy.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Args:
        Q (numpy.ndarray): The Q-table.
        state (int): The current state.
        epsilon (float): The epsilon to use for the calculation.

    Returns:
        int: The next action index.
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state])
    return action
