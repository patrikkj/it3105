import numpy as np
import random
from abc import abstractmethod
from copy import deepcopy

class StateManager:
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
    def get_observation(self):
        # NOTE: Observations must be hashable!
        ...

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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
