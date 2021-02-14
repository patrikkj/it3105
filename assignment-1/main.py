import matplotlib.pyplot as plt
import pandas as pd

import logger
from actor import Actor
from critic import Critic
from agent import ActorCriticAgent
from environment import PegEnvironment


config = {
    "n_episodes": 300,
    "reset_on_explore": True,

    "environment_type": PegEnvironment.TRIANGLE,
    "critic_type": Critic.NETWORK,

    "environment_params": {
        "board_size": 6,
        "holes": [(2, 2)],
    },
    "critic_params": {
        "layer_dims": (10, 15, 5, 1),   # Ignored if not using Network critc
        "alpha": 1e-14,                  # 1e-4 for Network, 0.3 for table
        "decay_rate": 0.9,
        "discount_rate": 0.99,
    },
    "actor_params": {
        "alpha": 0.15,
        "decay_rate": 0.9,
        "discount_rate": 0.99,
        "epsilon": 0.7,
        "epsilon_decay": 0.99,
    },
}
# decay: '5x5': 0.99, '6x6': 0.99, '7x7': 0.9995


def main():
    # Unpack configuration
    n_episodes = config["n_episodes"]
    reset_on_explore = config["reset_on_explore"]

    critic_type = config["critic_type"]
    evironment_type = config["environment_type"]

    environment_params = config["environment_params"]
    critic_params = config["critic_params"]
    actor_params = config["actor_params"]

    # Run experiment
    with PegEnvironment.from_type(evironment_type, **environment_params) as env:
        # Print initial board
        print(env.board)

        # Configure agent
        critic = Critic.from_type(
            critic_type, env, **critic_params, reset_on_explore=reset_on_explore
        )
        actor = Actor(env, **actor_params, reset_on_explore=reset_on_explore)
        agent = ActorCriticAgent(env, actor, critic)
        agent.set_callbacks(
            on_episode_end=logger.on_episode_end, on_step_end=logger.on_step_end
        )

        # Run experiments
        agent.run(n_episodes)

        # Collect logs
        df_episodes = pd.DataFrame(logger.episode_logs)
        df_steps = pd.DataFrame(logger.step_logs)

        # Plot progression, if any... hehe :)
        df_episodes["n_pegs_left"].plot()
        plt.show()


def debug():
    import cProfile

    with cProfile.Profile() as pr:
        main()
    pr.print_stats(sort=1)


main()
