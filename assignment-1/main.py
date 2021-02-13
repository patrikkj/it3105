import matplotlib.pyplot as plt
import pandas as pd

from actor import Actor
from agent import ActorCriticAgent
from critic import CriticNetwork, CriticTable
from peg_kernel import PegSolitaire
from plotpegs import plot

N_EPISODES = 5000
RESET_ON_EXPLORE = True

peg_params = {
    "board_type": "triangle",
    "board_size": 6,
    "holes": [(2, 2)],
}

critic_table_params = {
    "alpha": 0.3, 
    "decay_rate": 0.9, 
    "discount_rate": 0.99, 
}

critic_nn_params = {
    "layer_dims": (15, 10, 4, 1), 
    "alpha": 3e-2, 
    "decay_rate": 0.9,
    "discount_rate": 0.95,
}

critic_nn_params = {
    "layer_dims": (10, 15, 5, 1), 
    "alpha": 1e-4, 
    "decay_rate": 0.9,
    "discount_rate": 0.99,
}

# actor_params = {
#     "alpha": 0.3,
#     "decay_rate": 0.9,
#     "discount_rate": 0.99,
#     "epsilon": 0.0,
#     "epsilon_decay": 0.99
# }

actor_params = {
    "alpha": 0.15,
    "decay_rate": 0.9,
    "discount_rate": 0.99,
    "epsilon": 1, 
    "epsilon_decay": 0.99
}
#decay: '5x5': 0.99, '6x6': 0.99, '7x7': 0.9995


step_logs = []
def on_step_end(agent, episode, step):
    series = {}
    series["episode"] = episode
    series["step"] = step
    series["board"] = agent.env.board.copy()
    series["n_pegs_left"] = agent.env.get_pegs_left()
    series["peg_move_direction"] = agent.env._peg_move_direction
    series["peg_start_position"] = agent.env._peg_start_position
    series["peg_end_position"] = agent.env._peg_end_position
    step_logs.append(series)

episode_logs = []
def on_episode_end(agent, episode):
    series = {}
    series["episode"] = episode
    series["board"] = agent.env.board.copy()
    series["n_pegs_left"] = agent.env.get_pegs_left()
    episode_logs.append(series)


def main():
    with PegSolitaire(**peg_params) as environment:
        # Print initial board
        print(environment.board)

        # Configure agent
        critic = CriticTable(environment, **critic_table_params)
        #critic = CriticNetwork(environment, **critic_nn_params)
        actor = Actor(environment, **actor_params)
        agent = ActorCriticAgent(environment, actor, critic)
        agent.set_callbacks(on_episode_end=on_episode_end, on_step_end=on_step_end)

        # Run experiments
        agent.run(N_EPISODES)

        # Collect logs
        df_episodes = pd.DataFrame(episode_logs)
        df_steps = pd.DataFrame(step_logs)

        # Plot progression, if any... hehe :) 
        df_episodes["n_pegs_left"].plot()
        plt.show()

    
def debug():
    import cProfile
    with cProfile.Profile() as pr:
        main()
    pr.print_stats(sort=1)

main()
