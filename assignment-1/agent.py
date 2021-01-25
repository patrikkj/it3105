from abc import abstractmethod
from utils import SAP

class Agent:
    pass


class ActorCriticAgent(Agent):
    def __init__(self, env, actor, critic):
        self.env = env
        self.actor = actor
        self.critic = critic    # We should initialize critic with small random values

    def run(self, num_episodes, render=False, render_steps=False):
        for ep in range(num_episodes):
            self.episode(ep, render_steps=render_steps)
            if render:
                self.env.render()

    def episode(self, ep, render_steps=False):
        #if ep%200 == 0:
        #    print(ep)
        # Reset eilgibilities for actor and critic
        self.actor.reset_eligibility()
        self.critic.reset_eligibility()
        self.actor.set_episode(ep)

        # Initialize environment and fetch initial state
        self.env.reset()
        state = self.env.get_observation()
        action, is_exploring = self.actor(state)
        sap = SAP(state, action)

        while not self.env.is_terminal():
            # Apply action to environment
            state, reward, is_terminal = self.env.step(sap.action)

            # Evaluate state and action using actor and critic
            if not is_terminal:
                action, is_exploring = self.actor(state)
            error = self.critic(reward, sap.state, state)

            # Update eligibilities
            self.actor.update_all(sap, error, is_exploring)
            self.critic.update_all(sap.state, error, is_exploring)

            # Create sap for next iteration
            sap = SAP(state, action)
        if self.env.get_pegs_left() == 1:
            print(" -------------- WIN!!! -------------- ")
        else:
            print(f"Remaining: {self.env.get_pegs_left()}")
        #print(f"Episode {ep}, Pegs left: {self.env.get_pegs_left()}")
