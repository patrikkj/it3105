from utils import SAP
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class ActorCriticAgent:
    def __init__(self, env, actor, critic):
        self.env = env
        self.actor = actor
        self.critic = critic    # We should initialize critic with small random values
        # Callbacks for logging (self -> ...)
        self.on_episode_begin = []  # Format: (agent, episode) -> ...
        self.on_episode_end = []    # Format: (agent, episode) -> ...
        self.on_step_end = []       # Format: (agent, episode, step,) -> ...
        self._episode_results = []
        # Internal variables for book-keeping
        self._episode = 1
        self._step = 1
        self._training = False

    def set_callbacks(self, on_episode_begin=None,  on_episode_end=None, on_step_end=None):
        if on_episode_begin is not None:
            self.on_episode_begin = on_episode_begin
        if on_episode_end is not None:
            self.on_episode_end = on_episode_end
        if on_step_end is not None:
            self.on_step_end = on_step_end

    def run(self, num_episodes, training=True):
        self._episode_results = [{} for i in range(num_episodes)]
        for i in range(num_episodes):
            print("Episode:", i)       

            # Callbacks for episode start
            for callback in self.on_episode_begin:
                callback(self, self._episode)

            # Run episode
            self.episode(training=training)
        
            # Callbacks for episode end
            final = i == (num_episodes - 1)
            for callback in self.on_episode_end:
                callback(self, self._episode, final=final)
            self._episode_results[i]["best_x"] = (self.env.best_step, self.env.best_x)
            self._episode += 1
        self.show_heatmap()
        self.show_best_x()

    def fit(self, num_episodes):
        return self.run(num_episodes)

    def evaluate(self, num_episodes):
        return self.run(num_episodes, training=False)
        
    def episode(self, training=True): 
        print("enter episode",self._episode)
        # Reset eilgibilities for actor and critic
        self.actor.reset_eligibility()
        self.critic.reset_eligibility()
        self.actor.set_episode(self._episode)

        # Initialize environment and fetch initial state
        self.env.reset()
        obs = self.env.get_observation()
        state = self.env.decode_state(*obs)
        action, is_exploring, state_val = self.actor(state, *obs, training=training)
        sap = SAP(state, action)

        self._step = 1
        while not self.env.is_finished():
            print(f"Step {self._step} of episode {self._episode}")
            # Apply action to environment
            obs, reward, is_terminal = self.env.apply_action(sap.action)
            state = self.env.decode_state(*obs)
            # Evaluate state and action using actor and critic
            if not is_terminal:
                #TESTING
                print(f"    Getting actor action, state_val for x,v = {obs[0]}, {obs[1]}")
                action, is_exploring, state_val = self.actor(state, *obs, training=training)
            error = self.critic.td_error(reward, sap.state, state)

            # Update weights & eligibilities
            #TESTING
            #self.actor.update(sap, error, is_exploring)
            print("------------------------------------------------")
            print("     Update critic with error:", error)
            self.critic.update(sap.state, error, is_exploring)

            # Create sap for next iteration
            sap = SAP(state, action)

            # Callbacks
            for callback in self.on_step_end:
                callback(self, self._episode, self._step)
            self._step += 1

        # Used for logging
        self._training = training

    def show_best_x(self):
        plt.plot([ep["best_x"][1] for ep in self._episode_results])
        plt.show()
        return

    def show_heatmap(self):
        d = 100
        num_axis_ticks= 4
        heatmap_data = []
        x_vals = np.linspace(*self.env.x_range,d)
        v_vals = np.linspace(self.env.v_range[1],self.env.v_range[0],d)
        for v in v_vals:
            row = []
            for x in x_vals:
            


                state = self.env.decode_state(x,v)
                action, is_finished, state_val = self.actor(state, x, v, training=False)
                #TESTING
                #if not (isinstance(state_val, float) or isinstance(state_val, int)):
                    #print(type(state_val),state_val)

                    #state_val = float(state_val.numpy()[0][0])
                row.append(state_val)
            heatmap_data.append(row)
        sns.set_theme()
        #x_vals = [x for i,x in enumerate(x_vals) if i%10 == 0]
        x_axis = []
        y_axis = []
        rounding = 2
        for i in range(len(x_vals)):
            if (i+1) % (d/num_axis_ticks)==0:
                x_val = round(x_vals[i],rounding)
                y_val = round(v_vals[i],rounding)
            elif i ==0:
                x_val = round(x_vals[i],rounding)
                y_val = round(v_vals[i],rounding)
            else:
                x_val = None
                y_val = None
            x_axis.append(x_val)
            y_axis.append(y_val)
        print("X vals", x_vals)
        ax = sns.heatmap(heatmap_data, xticklabels=x_axis, yticklabels=y_axis)
        plt.show()

        
