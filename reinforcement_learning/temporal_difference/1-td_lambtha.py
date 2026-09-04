#!/usr/bin/env python3
"""
Module for performing the TD(lambtha) algorithm.
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambtha) algorithm.

    Args:
        env: The environment instance.
        V (numpy.ndarray): Array of shape (s,) containing the value
            estimate.
        policy (callable): Function that takes in a state and
            returns the next action to take.
        lambtha (float): The eligibility trace factor.
        episodes (int): The total number of episodes to train over.
        max_steps (int): The maximum number of steps per episode.
        alpha (float): The learning rate.
        gamma (float): The discount rate.

    Returns:
        numpy.ndarray: The updated value estimate V.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros_like(V)

        for step in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(
                action
            )

            delta = reward + gamma * V[new_state] - V[state]
            eligibility[state] += 1

            V = V + alpha * delta * eligibility
            eligibility = gamma * lambtha * eligibility

            state = new_state

            if terminated or truncated:
                break

    return V
