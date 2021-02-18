import json, graphics, utils, logger
import matplotlib.pyplot as plt
import pandas as pd
from actor import Actor
from agent import ActorCriticAgent
from critic import Critic
from environment import PegEnvironment
from utils import Cases


# Config IO handling
case = Cases.DIAMOND_TABLE_4.value
mode = "read"   # "read", "write" or ""

config = {
    "n_episodes": 2000,
    "reset_on_explore": True,

    "visualize_episodes": [-1],
    "delay" : 0.2,

    "environment_type": PegEnvironment.TRIANGLE,
    "critic_type": Critic.TABLE,

    "environment_params": {
        "board_size": 5,
        "holes": [(2, 2)], 
    },
    "critic_params": {
        "layer_dims": (10, 15, 5, 1),   # Ignored if not using Network critc
        "alpha": 0.3,                   # 1e-2 for Network, 0.3 for table
        "decay_rate": 0.9,
        "discount_rate": 0.99,
    },
    "actor_params": {
        "alpha": 0.15,
        "decay_rate": 0.9,
        "discount_rate": 0.99,
        "epsilon": 0.5,
        "epsilon_min": 0.03,
        "epsilon_decay": 0.99,
    },
}


def main():
    # Read/write configuration file
    if mode == "read":
        config = utils.read_config(case)
    elif mode == "write":
        utils.write_config(config, case)

    # Unpack configuration
    n_episodes = config["n_episodes"]
    reset_on_explore = config["reset_on_explore"]
    visualize_episodes = config["visualize_episodes"]
    delay = config["delay"]

    critic_type = config["critic_type"]
    environment_type = config["environment_type"]

    environment_params = config["environment_params"]
    critic_params = config["critic_params"]
    actor_params = config["actor_params"]

    # Run experiment
    with PegEnvironment.from_type(environment_type, **environment_params) as env:
        # Print initial board
        print(f"\n{'Initial board':^100}\n" + "="*100)
        print(env.board)

        # Print configuration
        print(f"\n{'Configuration':^100}\n" + "="*100)
        print(json.dumps(config, indent=4))

        # Configure agent
        critic = Critic.from_type(
            critic_type, env, **critic_params, reset_on_explore=reset_on_explore
        )
        actor = Actor(env, **actor_params, reset_on_explore=reset_on_explore)
        agent = ActorCriticAgent(env, actor, critic)
        agent.set_callbacks(
            on_episode_begin=[logger.step_logger],
            on_episode_end=[logger.episode_logger, logger.episode_reporter],
            on_step_end=[logger.step_logger]
        )

        # Run experiments
        print(f"\n{'Training':^100}\n" + "="*100)
        agent.fit(n_episodes)

        # Evaluate
        print(f"\n{'Evaluation':^100}\n" + "="*100)
        agent.evaluate(50)

        # Collect logs
        df_episodes = pd.DataFrame(logger.episode_logs)
        df_steps = pd.DataFrame(logger.step_logs)

        # Visualize
        df_episodes["n_pegs_left"].plot()
        for episode in visualize_episodes:
            graphics.Graphics(env, df_steps, delay).visualize_episode(episode=episode)
        plt.show()

main()  
