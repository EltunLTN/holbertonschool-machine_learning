#!/usr/bin/env python3
"""
Module for performing the Monte Carlo algorithm.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                 alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Args:
        env: The environment instance.
        V (numpy.ndarray): Array of shape (s,) containing the value
            estimate.
        policy (callable): Function that takes in a state and
            returns the next action to take.
        episodes (int): The total number of episodes to train over.
        max_steps (int): The maximum number of steps per episode.
        alpha (float): The learning rate.
        gamma (float): The discount rate.

    Returns:
        numpy.ndarray: The updated value estimate V.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(
                action
            )
            episode_data.append((state, reward))
            state = new_state

            if terminated or truncated:
                break

        episode_data = np.array(episode_data, dtype=int)
        G = 0
        for state, reward in episode_data[::-1]:
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])

    return V
