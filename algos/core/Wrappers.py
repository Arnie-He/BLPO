from __future__ import annotations
from functools import cached_property
from typing import Any, Callable, ClassVar, Dict, Generic, Optional, Tuple, Union, List

import chex
import jax
import jax.numpy as jnp
import jumanji.env
import numpy as np

import gymnax

import jumanji
from jumanji import specs, tree_utils
from jumanji.env import ActionSpec, Environment, Observation, State
from jumanji.types import TimeStep
from jumanji.wrappers import AutoResetWrapper

from environments import GYM_ENV_NAMES, JUMANJI_ENV_NAMES

"""Wrappers for Gymnax environments to be used in Gym."""

import copy

import chex
import gymnasium as gym
from gymnasium import core
from gymnasium.vector import utils
import jax.random
from gymnax.environments import environment
from gymnax.environments import spaces


class FlattenedJumanjiToGymnaxWrapper:
    def __init__(self, env):
        self.env = AutoResetWrapper(env)
        self.action_space = env.action_spec()
        self.observation_space = env.observation_spec()

    def reset(self, rng_key: jax.random.PRNGKey, env_params=None) -> Tuple[Any, Any]:
        
        state, timestep = jax.jit(self.env.reset)(rng_key)
        return jax.tree_util.tree_flatten(timestep.observation)[0], state

    def step(self, rng_key: jax.random.PRNGKey, state: Any, action: Any, env_params=None) -> Tuple[Any, Any, float, bool, dict]:
        # Use env_params to modify behavior if needed
        if env_params is not None:
            self.env_params = env_params
        # Jumanji's step does not require rng_key, but Gymnax provides it, so we accept it
        state, timestep = jax.jit(self.env.step)(state, action)
        return jax.tree_util.tree_flatten(timestep.observation)[0], state, timestep.reward, timestep.done, timestep.info
    
    def get_observation_shape(self):
        key = jax.random.PRNGKey(0)
        state, timestep  = jax.jit(self.env.reset)(key)
        return jax.tree_util.tree_flatten(timestep.observation)[0].shape

    def get_env_params(self):
        return self.env_params

    def render(self, state: Any, env_params=None):
        return self.env.render(state)


class GymnaxToGymWrapper(gym.Env[core.ObsType, core.ActType]):
    """Wrap Gymnax environment as OOP Gym environment."""

    def __init__(
        self,
        env: environment.Environment,
        params: Optional[environment.EnvParams] = None,
        seed: Optional[int] = None,
    ):
        """Wrap Gymnax environment as OOP Gym environment.


        Args:
            env: Gymnax Environment instance
            params: If provided, gymnax EnvParams for environment (otherwise uses
              default)
            seed: If provided, seed for JAX PRNG (otherwise picks 0)
        """
        super().__init__()
        self._env = copy.deepcopy(env)
        self.env_params = params if params is not None else env.default_params
        self.metadata.update(
            {
                "name": env.name,
                "render_modes": (
                    ["human", "rgb_array"] if hasattr(env, "render") else []
                ),
            }
        )
        self.rng: chex.PRNGKey = jax.random.PRNGKey(0)  # Placeholder
        self._seed(seed)
        _, self.env_state = self._env.reset(self.rng, self.env_params)

    @property
    def action_space(self):
        """Dynamically adjust action space depending on params."""
        return spaces.gymnax_space_to_gym_space(self._env.action_space(self.env_params))

    @property
    def observation_space(self):
        """Dynamically adjust state space depending on params."""
        return spaces.gymnax_space_to_gym_space(
            self._env.observation_space(self.env_params)
        )

    def _seed(self, seed: Optional[int] = None):
        """Set RNG seed (or use 0)."""
        self.rng = jax.random.PRNGKey(seed or 0)

    def step(
        self, action: core.ActType
    ) -> Tuple[core.ObsType, float, bool, bool, Dict[Any, Any]]:
        """Step environment, follow new step API."""
        self.rng, step_key = jax.random.split(self.rng)
        o, self.env_state, r, d, info = self._env.step(
            step_key, self.env_state, action, self.env_params
        )

        # jax.debug.print("Done: {d}, Info: {info}", d=d, info=info)

        return o, r, d, d, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[Any] = None,  # dict
    ) -> Tuple[core.ObsType, Any]:  # dict]:
        """Reset environment, update parameters and seed if provided."""
        if seed is not None:
            self._seed(seed)
        if options is not None:
            self.env_params = options.get(
                "env_params", self.env_params
            )  # Allow changing environment parameters on reset
        self.rng, reset_key = jax.random.split(self.rng)
        o, self.env_state = self._env.reset(reset_key, self.env_params)
        return o, {}

    def render(
        self, mode="human"
    ) -> Optional[Union[core.RenderFrame, List[core.RenderFrame]]]:
        """use underlying environment rendering if it exists, otherwise return None."""
        return getattr(self._env, "render", lambda x, y: None)(
            self.env_state, self.env_params
        )