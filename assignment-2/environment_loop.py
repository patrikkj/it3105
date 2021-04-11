import matplotlib.pyplot as plt

from base import LearningAgent


class EnvironmentLoop:
    """
    Creates an environment loop for a two-agent scenario 
    where agents alternate between taking turns.
    """
    def __init__(self, env, agent_1, agent_2, pid_1=1, pid_2=2, framerate=1):
        self.env = env
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.pid_1 = pid_1
        self.pid_2 = pid_2
        self.agents = (self.agent_1, self.agent_2)
        self.framerate = framerate
        self.current_pid = -1
        self.current_state = None
        self.pid_to_agent = {
            pid_1: self.agent_1,
            pid_2: self.agent_2
        }
    
    def train_agents(self):
        for agent in self.agents:
            if isinstance(agent, LearningAgent):
                agent.learn()

    def play_turn(self):
        agent = self.pid_to_agent[self.current_pid]
        callable_ = agent.handle_mouse_event if hasattr(agent, "handle_mouse_event") else None
        self.env.render(block=False, pause=1/self.framerate, close=False, callable_=callable_)
        action = agent.get_action(self.current_state)
        state, reward, is_terminal = self.env.move(action, self.current_pid)

        # Update env-loop state
        self.current_state = state
        self.current_pid = state[0]

    def play_game(self):
        self.env.reset()
        
        # Fetch starting Player ID
        self.current_state = self.env.get_initial_observation()
        self.current_pid = self.current_state[0]

        # Show initial board
        #self.env.render(block=True, pause=0, close=False, callable_=print)

        # Execute game loop
        while not self.env.is_finished():
            self.play_turn()
        
        # Show final board
        self.env.render(block=True, pause=0, close=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
            