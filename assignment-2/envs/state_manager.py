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
    def move(self, player, action):
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
        ...
    
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
