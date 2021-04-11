from agents import HumanHexAgent, HumanHexAgentV2, MCTSAgent, NaiveMCTSAgent, RandomAgent
from envs.hex import HexEnvironment, HexRenderer
from envs.nim import NimEnvironment
from mcts.tree import default_policy, tree_policy
from mcts.actor import MCTSActor
from mcts.learner import MCTSLearner
from network import ActorNetwork
from replay import ReplayBuffer
from environment_loop import EnvironmentLoop

configs = {
    "n_episodes": 2000,
    "reset_on_explore": True,

    "nim_params": {
        "N" : 87,
        "K" : 5
    },

    "hex_params": {
        "board_size": 4                  # The size (k) of the k x k Hex board, where 3 ≤ k ≤ 10.
    },

    "learner_params": {
        "n_episodes": 30,
        "n_simulations": 200,
        "save_interval": 50,
        "batch_size": 64
    },

    "network_params": {
        "alpha": 0.001,                  # Learning rate
        "layer_dims": (10, 5, 1),   # Num. of hidden layers
        "optimizer": 'adam',            # One of: 'adagrad', 'sgd', 'rmsprop', 'adam'
        "activation": 'relu',           # One of: 'linear', 'sigmoid', 'tanh', 'relu'
        "batch_size": 64
    },

    "topp_params": {
        "m": 4,     # Number of ANETs to be cached in preparation for a TOPP
        "g": 200    # Number of games to be played between agents during round-robin tournament
    },
}


"""
def main_hex():
    with HexEnvironment(**configs["hex_params"]) as env:
        #HexRenderer.plot(env.board)
        network = ActorNetwork(env, **configs["network_params"])
        replay_buffer = ReplayBuffer(buffer_size=512)
        agent = MCTSAgent(env, tree_policy=tree_policy, target_policy=default_policy, replay_buffer=replay_buffer, **configs["mcts_params"])
        agent.run()
"""

def main_hex():
    env = HexEnvironment(**configs["hex_params"])

    # Agents
    agent_2 = HumanHexAgentV2(env)
    #agent_1 = NaiveMCTSAgent(env, n_simulations=10000)
    
    network = ActorNetwork(env, **configs["network_params"])
    replay_buffer = ReplayBuffer()

    actor = MCTSActor(env, network)                             # Instance of 'Actor'
    learner = MCTSLearner(
        env=env,
        tree_policy=tree_policy,
        target_policy=default_policy,            # =actor
        network=network, 
        replay_buffer=replay_buffer,   # Instance of 'Learner'
        **configs["learner_params"])
    agent_1 = MCTSAgent(env, actor, learner, mode="mcts")                 # Instance of 'LearningAgent' <- 'Agent'

    
    with EnvironmentLoop(env, agent_1, agent_2, framerate=10) as loop:
        loop.train_agents()
        loop.play_game()

def main_nim():
    with NimEnvironment(**configs["nim_params"]) as env:
        print("\n\n\n")
        print(" Initial stones:", env.stones)
        mover = 1
        winner = 0
        print("---------------------          GAME START        -----------------------------")
        print("\n")
        while not env.is_finished():
            mover = (mover +1) % 2
            random_move = env.random_move()
            env.move(mover, random_move)
            print(env.players[mover], " , stones removed: ", random_move, " , Stones remaining: ", env.stones)
        
        winner = env.players[env.winner()]
        print("\n")
        print( "====================         GAME FINISHED        ===================")
        print("\n")
        print( "Winner: ", winner)
        print("\n")
        print("CONGRATULATIONS!!!")
        print("\n\n\n")


def debug_hex():
    import cProfile
    with cProfile.Profile() as pr:
        main_hex()
    pr.print_stats(sort=1)


# Sketchup of refactored component structures
"""
env = HexEnrivonment()                              # Instance of 'StateManager'
    spec = EnvironmentSpec()
    grid = HexGrid()
    renderer = HexRenderer()

network = Network()
replay_buffer = ReplayBuffer()

actor = MCTSActor(env, network)                             # Instance of 'Actor'
learner = MCTSLearner(env, actor, network, replay_buffer)   # Instance of 'Learner'
    root = MonteCarloNode()
    mct = MonteCarloTree(root)

mcts_agent = MCTSAgent(env, actor, learner)         # Instance of 'LearningAgent' <- 'Agent'
random_agent = RandomAgent(env)                     # Instance of 'Agent'
human_agent = HumanAgent(env)                       # Instance of 'Agent'
minimax_agent = MiniMaxAgent(env)                   # Instance of 'Agent' (haha, use this to train for TOPP?)

environment_loop = EnvironmentLoop(
    agent_1=mcts_agent,
    agent_2=random_agemt
):
    if isinstance(agent, LearningAgent):
        agent.learn()
    ...
"""


main_hex()
#debug_hex()
#main_nim()
