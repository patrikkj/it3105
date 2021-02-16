import matplotlib.pyplot as plt
import pandas as pd

import logger
from actor import Actor
from critic import Critic
from agent import ActorCriticAgent
from environment import PegEnvironment
from graphics import Graphics


config = {
    "n_episodes": 1000,
    "reset_on_explore": True,
    "delay" : 0.6,

    "environment_type": PegEnvironment.TRIANGLE,
    "critic_type": Critic.TABLE,

    "environment_params": {
        "board_size": 6,
        "holes": [(2,2)],
    },
    "critic_params": {
        "layer_dims": (10, 15, 5, 1),   # Ignored if not using Network critc
        "alpha": 1e-4,                  # 1e-4 for Network, 0.3 for table
        "decay_rate": 0.9,
        "discount_rate": 0.99,
    },
    "actor_params": {
        "alpha": 0.15,
        "decay_rate": 0.9,
        "discount_rate": 0.99,
        "epsilon": 0.7,
        "epsilon_min": 0,
        "epsilon_decay": 0.99,
    },
}
# decay: '5x5': 0.99, '6x6': 0.99, '7x7': 0.9995


def main():
    # Unpack configuration
    n_episodes = config["n_episodes"]
    reset_on_explore = config["reset_on_explore"]

    critic_type = config["critic_type"]
    environment_type = config["environment_type"]
    delay = config["delay"]

    environment_params = config["environment_params"]
    critic_params = config["critic_params"]
    actor_params = config["actor_params"]

    # Run experiment
    with PegEnvironment.from_type(environment_type, **environment_params) as env:
        # Print initial board
        print(env.board)

        # Configure agent
        critic = Critic.from_type(
            critic_type, env, **critic_params, reset_on_explore=reset_on_explore
        )
        actor = Actor(env, **actor_params, reset_on_explore=reset_on_explore)
        agent = ActorCriticAgent(env, actor, critic)
        agent.set_callbacks(
            on_episode_end=[logger.episode_logger, logger.episode_reporter_wrapper(freq=50)],
            on_step_end=[logger.step_logger]
        )

        # Run experiments
        print(f"\n{'Training':^80}\n" + "="*80)
        agent.run(n_episodes)

        # Evaluate
        print(f"\n{'Evaluation':^80}\n" + "="*80)
        agent.run(200, training=False)

        # Collect logs
        df_episodes = pd.DataFrame(logger.episode_logs)
        df_steps = pd.DataFrame(logger.step_logs)
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #    print(df_steps[0:20])        # Plot progression, if any... hehe :)
        df_episodes["n_pegs_left"].plot()
        graphics = Graphics()
        graphics.visualize_episode(df_steps, environment_type,df_steps.iloc[-1]["episode"], delay)
        plt.show()


def debug():
    import cProfile

    with cProfile.Profile() as pr:
        main()
    pr.print_stats(sort=1)


main()
