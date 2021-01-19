from abc import abstractmethod


class Agent:
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def run(self, num_episodes):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def episode(self):
        pass


class ActorCriticAgent(Agent):
    def __init__(self, env, actor, critic):
        super().__init__(env)
        self.actor = actor
        self.critic = critic

    def run(self, num_episodes):
        # Initialize critic with small random values
        self.critic.initialize()

        for ep in num_episodes:
            self.episode()

    def episode(self):
        # Reset eilgibilities for actor and critic
        self.actor.reset_eligibility()
        self.critic.reset_eligibility()

        # Initialize environment and fetch initial state
        self.env.reset()
        state = self.env.get_state()
        action = self.actor(state)

        reward, done = 0, False
        while not self.env.is_terminal(state):
            state, reward, done = self.step(action)

    def step(self, action):
        # Fetch state from environment
        state = self.env.get_state()
        new_state = self.env.step(action)
        reward = self.env.get_reward()

        # Evaluate state and action using actor and critic
        action_next = self.actor(state)
        self.actor.set_eligibility(state, action, 1)
        error = self.critic(reward, state, new_state)

        # Update eligibilities
        self.actor.update_all(error)
        self.critic.update_all(error)
        return new_state, reward
