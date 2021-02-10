import pandas as pd
import graphics

from actor import Actor
from agent import ActorCriticAgent
from critic import CriticTable, CriticNetwork
from peg_kernel import PegSolitaire
from plotpegs import plot

N_EPISODES = 20

peg_params = {
    "board_type": "diamond",
    "board_size": 5,
    "holes": [(2, 1)],
}

critic_table_params = {
    "alpha": 0.3, 
    "decay_rate": 0.9, 
    "discount_rate": 0.99, 
}

critic_nn_params = {
    "layer_dims": (15, 20, 30, 5, 1), 
    "alpha": 1e-3, 
    "decay_rate": 1e-3, 
    "discount_rate": 1e-3
}

actor_params = {
    "alpha": 0.15,
    "decay_rate": 0.9,
    "discount_rate": 0.99,
    "epsilon": 1, 
    "epsilon_decay": 0.9995
}


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


# decay: '5x5': 0.99, '6x6': 0.99, '7x7': 0.9995
def main():
    with PegSolitaire(**peg_params) as environment:
        critic = CriticTable(environment, **critic_table_params)
        #critic = CriticNetwork(environment, **critic_nn_params)
        actor = Actor(environment, **actor_params)

        # Configure agent
        agent = ActorCriticAgent(environment, actor, critic)
        agent.set_callbacks(on_episode_end=on_episode_end, on_step_end=on_step_end)

        # Run experiments
        agent.run(N_EPISODES, render=False, render_steps=False)

        # Collect logs
        df_episodes = pd.DataFrame(episode_logs)
        df_steps = pd.DataFrame(step_logs)

        print("DF_EPISODES")
        print(df_episodes)
        
        print("DF_STEPS")
        print(df_steps)

        """
        peg_left_list = []
        for ep in range(len(episode_logs)):
            serie = episode_logs.get(ep)
            peg_count = serie.get("n_pegs_left")
            peg_left_list.append((ep, peg_count))
        
        plot(peg_left_list)

        print(episode_logs)
        """

def debug():
    import cProfile
    with cProfile.Profile() as pr:
        main()
    pr.print_stats(sort=1)

main()




step_logs = {
    (1, 1): {
        "board": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
        "n_pegs_left": 8
    },
    (1, 2): {
        "board": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
        "n_pegs_left": 8
    },
    (1, 3): {
        "board": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
        "n_pegs_left": 8
    },
    (1, 4): {
        "board": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
        "n_pegs_left": 8
    },

    (2, 1): {
        "board": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
        "n_pegs_left": 8
    },
    (2, 2): {
        "board": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
        "n_pegs_left": 8
    },
    (2, 3): {
        "board": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
        "n_pegs_left": 8
    },
    (2, 4): {
        "board": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
        "n_pegs_left": 8
    }
}

