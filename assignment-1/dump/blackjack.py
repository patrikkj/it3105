import numpy as np
from . import env

def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card():
    return int(np.random.choice(deck))


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(env.Environment):
    def __init__(self):
        self.action_space = np.array([0, 1])
        self.observation_space = (32, 11, 2)
        self.reset()

    def step(self, action):
        return self._hit() if action else self._stick()

    def reset(self):
        """Resets the environment."""
        self.dealer = [draw_card()]
        self.player = [draw_card(), draw_card()]
        return self._get_obs()

    def is_terminal(self):
        """Determines whether the given state is a terminal state."""
        return np.prod(self.player) > 21 or np.prod(self.dealer) >= 17

    def _hit(self):
        self.player.append(draw_card())
        if is_bust(self.player):
            reward = -1.
        else:
            reward = 0.
        return self._get_obs(), reward

    def _stick(self):
        done = True
        while sum_hand(self.dealer) < 17:
            self.dealer.append(draw_card())
        reward = cmp(score(self.player), score(self.dealer))
        return self._get_obs(), reward

    def _get_obs(self):
        return self._tuple_to_numeric_state((sum_hand(self.player), self.dealer[0], usable_ace(self.player)))

    def __str__(self):
        return f"Player: {self.player}    Dealer: {self.dealer}"


env = BlackjackEnv()
env.step(1)
env.step(0)
print(env)
