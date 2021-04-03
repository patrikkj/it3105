from envs.hex import HexEnvironment
from envs.nim import NimEnvironment


configs = {
    "n_episodes": 2000,
    "reset_on_explore": True,

    "nim_params": {
        "N" : 87,
        "K" : 5
    },

    "hex_params": {
        "board_size": 5                  # The size (k) of the k x k Hex board, where 3 ≤ k ≤ 10.
    },

    "mcts_params": {
        "n_episodes": 5,
        "n_simulations": 5,
    },

    "network_params": {
        "alpha": 0.15,                  # Learning rate
        "layer_dims": (10, 15, 5, 1),   # Num. of hidden layers
        "optimizer": 'adam',            # One of: 'adagrad', 'sgd', 'rmsprop', 'adam'
        "activation": 'relu',           # One of: 'linear', 'sigmoid', 'tanh', 'relu'
        "batch_size": 32
    },

    "actor_params": {
        "decay_rate": 0.9,          
        "discount_rate": 0.99,
        "epsilon": 0.5,
        "epsilon_min": 0.03,
        "epsilon_decay": 0.99,
    },

    "topp_params": {
        "m": 4,     # Number of ANETs to be cached in preparation for a TOPP
        "g": 200    # Number of games to be played between agents during round-robin tournament
    },
}


def main_hex():
    with HexEnvironment(**configs["hex_params"]) as env:
        network = ActorNetwork(**configs["network_params"])
        actor = Actor(env, network, **configs["actor_params"])
        agent = MCTSAgent(env, actor, **configs["mcts_params"])
        # TODO: Fix stubs


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

main_hex()
main_nim()