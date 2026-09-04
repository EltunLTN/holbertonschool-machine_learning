#!/usr/bin/env python3
"""Module that implements the Monte Carlo algorithm."""
import numpy as np


def monte_carlo(
        env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
        gamma=0.99):
    """
    Perform the Monte Carlo algorithm to estimate a value function.

    Args:
        env: environment instance
        V (numpy.ndarray): array of shape (s,) containing the
            value estimate
        policy: function that takes in a state and returns the
            next action to take
        episodes (int): total number of episodes to train over
        max_steps (int): maximum number of steps per episode
        alpha (float): learning rate
        gamma (float): discount rate

    Returns:
        numpy.ndarray: V, the updated value estimate
    """
    for ep in range(episodes):
        state, _ = env.reset()
        episode = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        episode = np.array(episode, dtype=int)
        G = 0
        for state, reward in episode[::-1]:
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])

    return V
