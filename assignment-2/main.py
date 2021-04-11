from agents import (HumanHexAgent, HumanHexAgentV2, MCTSAgent, NaiveMCTSAgent,
                    RandomAgent)
from environment_loop import EnvironmentLoop
from envs.hex import HexEnvironment
from envs.nim import NimEnvironment

config = {
    "nim_params": {
        "N" : 87,
        "K" : 5
    },

    "hex_params": {
        "board_size": 5                  # The size (k) of the k x k Hex board, where 3 ≤ k ≤ 10.
    },

    "buffer_params": {
        "buffer_size": 512
    },

    "learner_params": {
        "n_episodes": 3,
        "n_simulations": 200,
        "save_interval": 50,
        "batch_size": 64
    },

    "network_params": {
        "alpha": 0.01,                      # Learning rate
        "layer_dims": (64, 32),             # Num. of hidden layers
        "optimizer": 'adam',                # One of: 'adagrad', 'sgd', 'rmsprop', 'adam'
        "activation": 'relu',               # One of: 'linear', 'sigmoid', 'tanh', 'relu'
        "loss": 'categorical_crossentropy', # One of: 'categorical_crossentropy', 'kl_divergence'
        "batch_size": 64,
        "epochs": 5
    },

    "topp_params": {
        "m": 4,     # Number of ANETs to be cached in preparation for a TOPP
        "g": 200    # Number of games to be played between agents during round-robin tournament
    },
}


def main_hex():
    with HexEnvironment(**config["hex_params"]) as env:
        agent_1 = MCTSAgent.from_config(env, config)
        #agent_1 = NaiveMCTSAgent(env)
        #agent_1 = RandomAgent(env)
        agent_2 = RandomAgent(env)
        with EnvironmentLoop(env, agent_1, agent_2, framerate=20) as loop:
            loop.train_agents()
            loop.play_game()

def main_nim():
    with NimEnvironment(**config["nim_params"]) as env:
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


main_hex()
#main_nim()


