import numpy as np
import tensorflow as tf
import yaml
import os
#sys.path.append("/Users/patrikkj/git/patrikkj/it3105/")
os.chdir("/Users/patrikkj/git/patrikkj/it3105")
from agents import (HumanHexAgent, HumanHexAgentV2, MCTSAgent, NaiveMCTSAgent,
                    RandomAgent)
from environment_loop import EnvironmentLoop
from envs.hex import HexEnvironment
from tournament import Tournament
from utils import debug

tf.random.set_seed(0)
np.random.seed(0)


# Load configurations
with open("./assignment-2/config.yml") as f:
   config = yaml.load(f, Loader=yaml.FullLoader)

# with open("./assignment-2/config_topp.yml") as f:
#    config = yaml.load(f, Loader=yaml.FullLoader)

@debug
def main():

    ############################
    #   PRELIMINARY TESTING    #
    ############################

    # Illustration game between a Naíve MCTS and a random agent
    if False:
        with HexEnvironment(**config["hex_params"]) as env:
            agent_1 = NaiveMCTSAgent(env, n_simulations=10_000)
            agent_2 = HumanHexAgentV2(env)
            EnvironmentLoop(env, agent_1, agent_2, framerate=10).play_game()
    
    # Test a few pivotal parameters
    if True:
        with HexEnvironment(**config["hex_params"]) as env:
            agent_1 = MCTSAgent.from_checkpoint(
                env, 
                config["export_dir"], 
                name="candidate_oht", 
                episode=420).learn(episodes=20)
            
            #agent_1 = MCTSAgent.from_config(env, config).learn()
            agent_2 = HumanHexAgentV2(env)
            while input() != "quit":
                EnvironmentLoop(env, agent_1, agent_2, framerate=10).play_game()


    ###################
    #   TOURNAMENT    #
    ###################
    
    # Load progressive policies from directory and plays tournament
    if False:
        with HexEnvironment(**config["hex_params"]) as env:
            agents = MCTSAgent.from_agent_directory(env=env, export_dir=config["export_dir"])
            Tournament(env, agents, **config["topp_params"]).play_tournament()
    

if __name__ == "__main__":
    main()
