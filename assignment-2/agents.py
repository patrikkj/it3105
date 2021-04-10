
import random
from string import ascii_uppercase

import mcts.agent

from .base import Agent

MCTSAgent = mcts.agent.MCTSAgent


class RandomAgent(Agent):
    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        actions = self.env.get_legal_actions()
        return random.choice(list(actions))


class HumanHexAgent(Agent):
    def __init__(self, env):
        self.env = env

    def _fetch_and_decode(self):
        try:
            action = input(f"Select an action: ")
            alpha, num_str = action[0], action[1:]
            n = self.env.board_size
            row = ascii_uppercase.index(alpha.upper())
            col = int(num_str)
            assert 0 <= row < n
            assert 0 <= col < n
            return row * self.env.board_size + col 
        except:
            return -1

    def get_action(self, state):

        # Render environment for human to make a move
        self.env.render(block=True)
        legal_actions = self.env.get_legal_actions()

        # Ask for input
        player, *_ = state
        print(f"Your turn to make a move (you are player {player})!")
        while (action := self._fetch_and_decode()) not in legal_actions:
            print("    Oops, illegal action! Try again.")
        return action
