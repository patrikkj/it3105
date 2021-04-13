import json, utils, logger
import matplotlib.pyplot as plt
import pandas as pd
from actor import Actor
from agent import ActorCriticAgent
from critic import Critic
from utils import Cases
from mountain_car_env import MountainCar
from animate import AnimateMC

"""
# Config IO handling
case = Cases.DIAMOND_TABLE_4.value
mode = "read"   # "read", "write" or ""
"""

config = {
    "environment_params":{
        "x_range":[-1.2,0.6], 
        "v_range": [-0.07,0.07],
        "max_steps":1000
    },

    "tiling_params":{
        "n_tiles": 4,
        "n_tilings": 5,
        "displacement_vector":[1,3]
    },
    "n_episodes": 10,
    "reset_on_explore": True,

    "visualize_episodes": [-1],
    "delay" : 0.2,

    "critic_type": Critic.NETWORK,

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
        "epsilon_decay": 0.99
    }

}  
"""
# Read/write configuration file
if mode == "read":
    config = utils.read_config(case)
elif mode == "write":
    utils.write_config(config, case)
"""
def main():

    # Unpack configuration

    environment_params = config["environment_params"]
    tiling_params = config["tiling_params"]
    n_episodes = config["n_episodes"]
    reset_on_explore = config["reset_on_explore"]
    visualize_episodes = config["visualize_episodes"]
    delay = config["delay"]

    critic_type = config["critic_type"]

    critic_params = config["critic_params"]
    actor_params = config["actor_params"]

    # Run experiment
    env = MountainCar(**environment_params)
    env.init_tilings(environment_params["x_range"], environment_params["v_range"],**tiling_params)
    # Print configs
    print("Max steps before timeout: ", environment_params["max_steps"])

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
    animation = AnimateMC()
    

main()  
