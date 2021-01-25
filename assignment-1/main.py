from actor import Actor
from agent import ActorCriticAgent
from critic import CriticTable
from envs.peg_kernel import PegSolitaire


N_EPISODES = 1000

peg_params = {
    "board_type": "square",
    "board_size": 6,
    "holes": [(2, 2)],
}

critic_table_params = {
    "alpha": 0.15, 
    "decay_rate": 0.9, 
    "discount_rate": 0.99, 
}

actor_params = {
    "alpha": 0.15,
    "decay_rate": 0.9, 
    "discount_rate": 0.99, 
    "epsilon": 1, 
    "epsilon_decay": 0.99
}
# decay: '5x5': 0.99, '6x6': 0.99, '7x7': 0.9995
def main():
    with PegSolitaire(**peg_params) as environment:
        critic = CriticTable(environment, **critic_table_params)
        actor = Actor(environment, **actor_params)
        agent = ActorCriticAgent(environment, actor, critic)
        agent.run(N_EPISODES, render_steps=True)

def debug():
    import cProfile
    with cProfile.Profile() as pr:
        main()
    pr.print_stats(sort=1)

main()
#debug()

critic_nn_params = {
    "layer_dims": (15, 20, 30, 5, 1), 
    "alpha": 1e-3, 
    "decay_rate": 1e-3, 
    "discount_rate": 1e-3
}