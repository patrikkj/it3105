import pandas as pd

from actor import Actor
from agent import ActorCriticAgent
from critic import CriticTable
from peg_kernel import PegSolitaire

N_EPISODES = 1000

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

actor_params = {
    "alpha": 0.15,
    "decay_rate": 0.9,
    "discount_rate": 0.99,
    "epsilon": 1, 
    "epsilon_decay": 0.9995
}

episode_logs = {}
def on_episode_end(agent, episode):
    series = pd.Series()
    series["board"] = agent.env.board.copy()
    series["n_pegs_left"] = agent.env.get_pegs_left()
    episode_logs[episode] = series

stepZ_logs = {}
def on_step_end(agent, step):
    series = pd.Series()
    series["board"] = agent.env.board.copy()
    series["n_pegs_left"] = agent.env.get_pegs_left()
    step_logs[step] = series



# decay: '5x5': 0.99, '6x6': 0.99, '7x7': 0.9995
def main():
    with PegSolitaire(**peg_params) as environment:
        critic = CriticTable(environment, **critic_table_params)
        actor = Actor(environment, **actor_params)
        agent = ActorCriticAgent(environment, actor, critic)
        agent.set_callbacks(on_episode_end=on_episode_end)
        agent.run(N_EPISODES, render=False, render_steps=False)
        print(episode_logs)

def debug():
    import cProfile
    with cProfile.Profile() as pr:
        main()
    pr.print_stats(sort=1)

main()

critic_nn_params = {
    "layer_dims": (15, 20, 30, 5, 1), 
    "alpha": 1e-3, 
    "decay_rate": 1e-3, 
    "discount_rate": 1e-3
}
