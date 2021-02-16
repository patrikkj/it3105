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
        self.on_step_end = []       # Format: (agent, episode, step) -> ...
        self._current_episode = 0

    def set_callbacks(self, on_episode_begin=None,  on_episode_end=None, on_step_end=None):
        if on_episode_begin is not None:
            self.on_episode_begin = on_episode_begin
        if on_episode_end is not None:
            self.on_episode_end = on_episode_end
        if on_step_end is not None:
            self.on_step_end = on_step_end

    def run(self, num_episodes, training=True):
        for _ in range(num_episodes):
            # Callbacks for episode start
            for callback in self.on_episode_begin:
                callback(self, self._current_episode)

            # Run episode
            self.episode(training=training)
        
            # Callbacks for episode end
            for callback in self.on_episode_end:
                callback(self, self._current_episode)
        
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

        step = 0
        while not self.env.is_terminal():
            # Apply action to environment
            state, reward, is_terminal = self.env.step(sap.action)

            # Evaluate state and action using actor and critic
            if not is_terminal:
                action, is_exploring = self.actor(state, training=training)
            error = self.critic.td_error(reward, sap.state, state)

            # Update weights & eligibilities
            self.actor.update(sap, error, is_exploring)
            self.critic.update(sap.state, error, is_exploring)

            # Create sap for next iteration
            sap = SAP(state, action)

            # Callbacks
            for callback in self.on_step_end:
                callback(self, self._current_episode, step)
            step += 1

        #if self._current_episode%50 == 0:
        #    tf.print(f"Episode: {self._current_episode} \teps: {self.actor.epsilon * self.actor.epsilon_decay ** self.actor.episode}")
        
        # Some printing :)
        #if self.env.get_pegs_left() == 1:
        #    tf.print(" -------------- WIN!!! -------------- ")
        #else:
        #    tf.print(f"Remaining: {self.env.get_pegs_left()}")

        #else:
        #    tot_pegs = self.env._board.sum()
        #    cur_pegs = self.env.get_pegs_left()
        #    print(f"[{'#'*int(100*((tot_pegs-cur_pegs+1)/tot_pegs)):<100}]")

        #print(f"Episode {ep}, Pegs left: {self.env.get_pegs_left()}")
