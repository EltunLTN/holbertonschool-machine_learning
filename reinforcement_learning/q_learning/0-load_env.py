#!/usr/bin/env python3
"""
Module for loading the FrozenLakeEnv environment from gymnasium.
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.

    Args:
        desc (list of lists): Custom description of the map to
            load for the environment. If None, either map_name is
            used or a random map is generated.
        map_name (str): The pre-made map to load. If None and desc
            is None, a randomly generated 8x8 map is loaded.
        is_slippery (bool): Determines if the ice is slippery.

    Returns:
        The gymnasium FrozenLake environment.
    """
    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode='ansi'
    )
    return env
