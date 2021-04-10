import random

from ..base import StateManager


class NimEnvironment(StateManager):
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
