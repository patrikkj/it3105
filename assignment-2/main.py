import numpy as np
import tensorflow as tf
import yaml

from agents import (HumanHexAgent, HumanHexAgentV2, MCTSAgent, NaiveMCTSAgent,
                    RandomAgent)
from environment_loop import EnvironmentLoop
from envs.hex import HexEnvironment
from tournament import Tournament
from utils import debug

tf.random.set_seed(1)
np.random.seed(1)

# Load configuration
CONFIG_PATH = "./assignment-2/config_topp.yml"   # config.yml / config_topp.json
with open(CONFIG_PATH) as f:
   config = yaml.load(f, Loader=yaml.FullLoader)


def main():
    with HexEnvironment(**config["hex_params"]) as env:
        # Agents
        #random = RandomAgent(env)
        #human = HumanHexAgentV2(env)
        #mcts = MCTSAgent.from_config(env, config).learn()
        #mcts_ckpt = MCTSAgent.from_checkpoint(env, config["export_dir"], episode=100)

        # Test a few pivotal parameters
        if False:
            agent_1 = MCTSAgent.from_config(env, config).learn()
            agent_2 = HumanHexAgentV2(env)
            EnvironmentLoop(env, agent_1, agent_2, framerate=10).play_game()

        # Train a set of agents and run a short tournament (use 'config_topp.json')
        if False:
            # Train agent and play against it
            agent_1 = MCTSAgent.from_config(env, config).learn()
            agent_2 = HumanHexAgentV2(env)
            EnvironmentLoop(env, agent_1, agent_2, framerate=10).play_game()
        
        # Load progressive policies from directory and run tournament
        if True:
            agents = MCTSAgent.from_agent_directory(env=env, export_dir=config["export_dir"])
            Tournament(env, agents, **config["topp_params"]).play_tournament()
    

if __name__ == "__main__":
    main()
