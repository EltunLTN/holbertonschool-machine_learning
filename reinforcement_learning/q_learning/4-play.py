#!/usr/bin/env python3
"""
Module for having a trained agent play an episode of FrozenLake.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode, always exploiting the
    Q-table.

    Args:
        env: The FrozenLakeEnv instance.
        Q (numpy.ndarray): The Q-table.
        max_steps (int): The maximum number of steps in the
            episode.

    Returns:
        tuple: (total_rewards, rendered_outputs)
            total_rewards (float): The total rewards for the
                episode.
            rendered_outputs (list): A list of rendered outputs
                (str) representing the board state at each step,
                including the final state after the episode ends.
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = []

    rendered_outputs.append(env.render())

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, terminated, truncated, _ = env.step(
            action
        )
        total_rewards += reward

        action_names = ['Left', 'Down', 'Right', 'Up']
        rendered_outputs[-1] += f'  ({action_names[action]})'
        rendered_outputs.append(env.render())

        state = new_state

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
