import numpy as np
import random
from abc import abstractmethod

class Environment:
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



class NimEnvironment(Environment):
    REWARD_WIN = 500
    REWARD_ACTION = 0
    REWARD_LOSS = - 100


    def __init__(self, N=45, K=5):
        self.N = N
        self.K = K
        self.players = {0: "Player 1" , 1: "Player 2"}
        self.last_mover = -1;
        self.stones = N

    
    def move (self, player, action):
        if not self.legal_move(player, action):
            return False
        self.stones -= action
        self.last_mover = (self.last_mover + 1) % 2

    def is_finished(self):
        return self.stones <= 0

    def winner(self):
        if not self.is_finished:
            return None
        else:
            return self.last_mover

    def legal_move(self, player, action):
        if player == self.last_mover or action > self.K or action < 1:
            return False
        return True

    def random_move(self):
        max = self.get_legal_actions()
        return random.randint(1, max)

    def get_legal_actions(self):
        return min(self.K, self.stones)