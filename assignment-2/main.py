from envs.hex import HexEnvironment, HexRenderer
from envs.nim import NimEnvironment
from actors.network import ActorNetwork
from agents.mcts import MCTSAgent, tree_policy, default_policy
from agents.replay import ReplayBuffer

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

    "mcts_params": {
        "n_episodes": 300,
        "n_simulations": 200,
        "save_interval": 50,
        "batch_size": 64
    },

    "network_params": {
        "alpha": 0.15,                  # Learning rate
        "layer_dims": (10, 15, 5, 1),   # Num. of hidden layers
        "optimizer": 'adam',            # One of: 'adagrad', 'sgd', 'rmsprop', 'adam'
        "activation": 'relu',           # One of: 'linear', 'sigmoid', 'tanh', 'relu'
        "batch_size": 32
    },

    "topp_params": {
        "m": 4,     # Number of ANETs to be cached in preparation for a TOPP
        "g": 200    # Number of games to be played between agents during round-robin tournament
    },
}



def main_hex():
    with HexEnvironment(**configs["hex_params"]) as env:
        #HexRenderer.plot(env.board)
        network = ActorNetwork(env, **configs["network_params"])
        replay_buffer = ReplayBuffer(buffer_size=512)
        agent = MCTSAgent(env, tree_policy=tree_policy, target_policy=default_policy, replay_buffer=replay_buffer, **configs["mcts_params"])
        agent.run()


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


#main_hex()
debug_hex()
#main_nim()
