#!/usr/bin/env python3
"""
Module for performing Q-learning on the FrozenLakeEnv environment.
"""
import numpy as np


epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning.

    Args:
        env: The FrozenLakeEnv instance.
        Q (numpy.ndarray): The Q-table.
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
        tuple: (Q, total_rewards)
            Q (numpy.ndarray): The updated Q-table.
            total_rewards (list): A list containing the rewards
                per episode.
    """
    total_rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(
                action
            )

            if terminated and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) -
                Q[state, action]
            )

            state = new_state
            episode_reward += reward

            if terminated or truncated:
                break

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

        total_rewards.append(episode_reward)

    return Q, total_rewards
