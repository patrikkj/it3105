from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class EnvironmentSpec:
    observations: int
    actions: int

class StateManager:
    def __init__(self, spec):
        self._spec = spec

    @abstractmethod
    def move(self, action, player):
        """
        Should return the next state, reward 
        and whether the new state is a terminal state.
        """ 
        ...

    @abstractmethod
    def reset(self):
        ...
    
    @abstractmethod
    def is_finished(self):
        ...
        
    @abstractmethod
    def get_legal_actions(self):
        ...

    @abstractmethod
    def get_initial_observation(self):
        ...

    @abstractmethod
    def get_observation(self):
        # NOTE: Observations must be hashable!
        ...

    @abstractmethod
    def get_winner(self):
        ...

    @property
    def spec(self):
        """Describes the specifications of the managed environment.
        Should be instance of EnvironmentSpec.
        """
        return self._spec

    def copy(self):
        """
        Returns a duplicate of the current environment.
        NOTE: Should be overwritten, deepcopy can be slow for nested structures!
        """
        return deepcopy(self)

    def decode_state(self, state):
        """
        Overload this method if 'self.get_observation()' 
        returns a compressed state representation.
        """
        raise state

    def render(self, block=True, pause=0.1, close=True):
        """
        Renders the current environment.
        Setting 'block=True' should cause the rendering backend to block execution.
        Setting 'pause' controls how long execution will be blocked.
        Setting 'close=True' should automatically close the visualization when execution continues.
        """
        raise NotImplementedError

    @staticmethod
    def apply(state, action):
        """
        Returns the state defined by applying 'action' to 'state'.
        Convenience function, does no perform any move validation
        nor changes to the state of the environment.
        """
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

class Actor:
    @abstractmethod
    def get_action(self, state):
        ...

class Learner:
    @abstractmethod
    def learn(self):
        ...
    
    @abstractmethod
    def step(self):
        ...

class Agent:
    @abstractmethod
    def get_action(self, state):
        ...

    def update(self):
        """
        Agents can optionally override this method to perform learning
        on the agents' underlying actor.
        """
        ...

class LearningAgent(Agent):
    @abstractmethod
    def learn(self):
        ...

    @abstractmethod
    def save(self):
        ...

    @abstractmethod
    def load(self):
        ...
