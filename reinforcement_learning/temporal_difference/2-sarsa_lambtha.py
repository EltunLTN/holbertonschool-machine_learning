#!/usr/bin/env python3
"""
Module for performing the SARSA(lambtha) algorithm.
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


def sarsa_lambtha(
        env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
        gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(lambtha) algorithm.

    Args:
        env: The environment instance.
        Q (numpy.ndarray): Array of shape (s, a) containing the
            Q table.
        lambtha (float): The eligibility trace factor.
        episodes (int): The total number of episodes to train over.
        max_steps (int): The maximum number of steps per episode.
        alpha (float): The learning rate.
        gamma (float): The discount rate.
        epsilon (float): The initial threshold for epsilon greedy.
        min_epsilon (float): The minimum value that epsilon should
            decay to.
        epsilon_decay (float): The decay rate for updating epsilon
            between episodes.

    Returns:
        numpy.ndarray: The updated Q table.
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros_like(Q)

        for step in range(max_steps):
            new_state, reward, terminated, truncated, _ = env.step(
                action
            )
            new_action = epsilon_greedy(Q, new_state, epsilon)

            delta = (reward + gamma * Q[new_state, new_action] -
                     Q[state, action])
            eligibility[state, action] += 1

            Q = Q + alpha * delta * eligibility
            eligibility = gamma * lambtha * eligibility

            state = new_state
            action = new_action

            if terminated or truncated:
                break

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q
