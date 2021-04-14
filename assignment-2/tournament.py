from dataclasses import dataclass
from itertools import combinations

from base import Agent


@dataclass
class Participant:
    _tid = 0    # Class-level attribute for generating tournament ID's
    
    agent: Agent
    tid: int
    wins: int = 0
    losses: int = 0

    @property
    def winrate(self):
        return self.wins / (self.wins + self.losses)

    @classmethod
    def from_agent(cls, agent):
        Participant._tid += 1
        return Participant(agent=agent, tid=Participant._tid)


class Tournament:
    """
    Plays a round-robin tournament between attending agents.
        PID = Player ID
        TID = Participant (tournament) ID
    """
    def __init__(self, env, agents, num_series=25, framerate=1):
        self.env = env
        self.participants = [Participant.from_agent(agent) for agent in agents]
        self.num_series = num_series
        self.framerate = framerate

    @staticmethod
    def alternate(a, b, reverse=False):
        if reverse:
            a, b = b, a
        while True:
            yield (a, 1)
            yield (b, 2)
    
    @staticmethod
    def round_robin(participants):
        return list(combinations(participants, 2))

    def play_game(self, agent_1, agent_2, reverse=False):
        self.env.reset()
        
        # Fetch starting Player ID and generate play order
        participant_gen = Tournament.alternate(agent_1, agent_2, reverse=reverse)
        state = self.env.get_initial_observation()

        # Execute game loop
        while not self.env.is_finished():
            participant, pid = next(participant_gen)
            action = participant.agent.get_action(state)
            state, _, _ = self.env.move(action, pid)
            
        # Assign winner
        participant.wins += 1
        next(participant_gen)[0].losses += 1

    def play_tournament(self):
        matches = Tournament.round_robin(self.participants)
        for participant_1, participant_2 in matches:
            for game in range(self.num_series):
                reverse = bool(game % 2)
                self.play_game(participant_1, participant_2, reverse=reverse)
        self.print_summary()

    def print_summary(self):
        # Sort standings by value
        participants = sorted(self.participants, key=lambda p: p.winrate, reverse=True)

        print("\n ======= Tournament ======= ")
        for i, p in enumerate(participants, start=1):
            print(f" ({i})    Agent {p.tid:<4}  W/L: {f'{p.wins}/{p.losses}':<6}  ({p.winrate*100:.2f}%)")
        
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
            