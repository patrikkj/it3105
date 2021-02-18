from utils import SAP
import tensorflow as tf
import numpy as np


class ActorCriticAgent:
    def __init__(self, env, actor, critic):
        self.env = env
        self.actor = actor
        self.critic = critic    # We should initialize critic with small random values

        # Callbacks for logging (self -> ...)
        self.on_episode_begin = []  # Format: (agent, episode) -> ...
        self.on_episode_end = []    # Format: (agent, episode) -> ...
        self.on_step_end = []       # Format: (agent, episode, step,) -> ...

        # Internal variables for book-keeping
        self._current_episode = 0
        self._training = False
        self._error = 0

    def set_callbacks(self, on_episode_begin=None,  on_episode_end=None, on_step_end=None):
        if on_episode_begin is not None:
            self.on_episode_begin = on_episode_begin
        if on_episode_end is not None:
            self.on_episode_end = on_episode_end
        if on_step_end is not None:
            self.on_step_end = on_step_end

    def run(self, num_episodes, training=True):
        # Used for logging purposes
        self._final_episode = num_episodes

        for i in range(num_episodes):
            # Callbacks for episode start
            for callback in self.on_episode_begin:
                callback(self, self._current_episode)

            # Run episode
            self.episode(training=training)
        
            # Callbacks for episode end
            final = i == (num_episodes - 1)
            for callback in self.on_episode_end:
                callback(self, self._current_episode, final=final)

    def fit(self, num_episodes):
        return self.run(num_episodes)

    def evaluate(self, num_episodes):
        return self.run(num_episodes, training=False)
        
    def episode(self, training=True):
        self._training = training
        self._current_episode += 1
        
        # Reset eilgibilities for actor and critic
        self.actor.reset_eligibility()
        self.critic.reset_eligibility()
        self.actor.set_episode(self._current_episode)

        # Initialize environment and fetch initial state
        self.env.reset()
        state = self.env.get_observation()
        action, is_exploring = self.actor(state, training=training)
        sap = SAP(state, action)

        step = 1
        while not self.env.is_terminal():
            # Apply action to environment
            state, reward, is_terminal = self.env.step(sap.action)

            # Evaluate state and action using actor and critic
            if not is_terminal:
                action, is_exploring = self.actor(state, training=training)
            error = self.critic.td_error(reward, sap.state, state)
            self._error = float(error)

            # Update weights & eligibilities
            self.actor.update(sap, error, is_exploring)
            self.critic.update(sap.state, error, is_exploring)

            # Create sap for next iteration
            sap = SAP(state, action)

            # Callbacks
            for callback in self.on_step_end:
                callback(self, self._current_episode, step)
            step += 1
