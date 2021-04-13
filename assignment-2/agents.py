
import random
from string import ascii_uppercase

import matplotlib.pyplot as plt

import mcts.agent
from base import Agent

MCTSAgent = mcts.agent.MCTSAgent
NaiveMCTSAgent = mcts.agent.NaiveMCTSAgent

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
            y = ascii_uppercase.index(alpha.upper())
            x = int(num_str) - 1
            assert 0 <= y < n
            assert 0 <= x < n
            return x * self.env.board_size + y 
        except:
            return -1

    def get_action(self, state):
        # Render environment for human to make a move
        legal_actions = self.env.get_legal_actions()

        # Ask for input
        player, *_ = state
        print(f"\nPlayer {player}'s turn!")
        while (action := self._fetch_and_decode()) not in legal_actions:
            print("    Oops, illegal action! Try again.")
        return action


class HumanHexAgentV2(Agent):
    action = None

    def __init__(self, env):
        self.env = env
    
    @staticmethod
    def handle_mouse_event(event):
        HumanHexAgentV2.action = event.artist.action

    def get_action(self, state):
        print("Click on hexagon to move ...")
        legal_actions = self.env.get_legal_actions()
        while (action := HumanHexAgentV2.action) not in legal_actions:
            plt.waitforbuttonpress()
        return action

# TODO: MiniMaxAgent? (haha, use this to train for TOPP?)
