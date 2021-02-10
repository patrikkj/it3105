from abc import abstractmethod
from utils import SAP

class Agent:
    pass


class ActorCriticAgent(Agent):
    def __init__(self, env, actor, critic):
        self.env = env
        self.actor = actor
        self.critic = critic    # We should initialize critic with small random values

        # Callbacks for logging (self -> ...)
        self.on_episode_end = None
        self.on_step_end = None

    def set_callbacks(self, on_episode_end=None, on_step_end=None, episode_callback_freq=200):
        if on_episode_end:
            self.on_episode_end = on_episode_end
        if on_step_end:
            self.on_step_end = on_step_end
        self.episode_callback_freq = episode_callback_freq

    def run(self, num_episodes, render=False, render_steps=False):
        for ep in range(num_episodes):
            self.episode(ep, render_steps=render_steps)
        
    def episode(self, ep, render=False, render_steps=False):
        # Reset eilgibilities for actor and critic
        self.actor.reset_eligibility()
        self.critic.reset_eligibility()
        self.actor.set_episode(ep)

        # Initialize environment and fetch initial state
        self.env.reset()
        state = self.env.get_observation()
        action = self.actor(state)
        sap = SAP(state, action)

        step = 0
        while not self.env.is_terminal():
            # Apply action to environment
            state, reward, is_terminal = self.env.step(sap.action)

            # Evaluate state and action using actor and critic
            if not is_terminal:
                action = self.actor(state)
            error = self.critic(reward, sap.state, state)

            # Update eligibilities
            self.actor.update_all(sap, error)
            self.critic.update_all(sap.state, error)

            # Create sap for next iteration
            sap = SAP(state, action)

            # Callbacks
            if render_steps:
                self.env.render()
            if self.on_step_end:
                self.on_step_end(self, ep, step)
            step += 1
        
        # Callbacks
        if render:
            self.env.render()
        if self.on_episode_end:
            self.on_episode_end(self, ep)

        # Some prinnting :)
        if self.env.get_pegs_left() == 1:
            print(" -------------- WIN!!! -------------- ")
        else:
            print(f"Remaining: {self.env.get_pegs_left()}")

        #else:
        #    tot_pegs = self.env._board.sum()
        #    cur_pegs = self.env.get_pegs_left()
        #    print(f"[{'#'*int(100*((tot_pegs-cur_pegs+1)/tot_pegs)):<100}]")

        #print(f"Episode {ep}, Pegs left: {self.env.get_pegs_left()}")
