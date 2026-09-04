#!/usr/bin/env python3
"""
Module for initializing the Q-table.
"""
import numpy as np


def q_init(env):
    """
    Initializes the Q-table.

    Args:
        env: The FrozenLakeEnv instance.

    Returns:
        numpy.ndarray: The Q-table initialized as zeros, with
            shape (number of states, number of actions).
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    return np.zeros((n_states, n_actions))
