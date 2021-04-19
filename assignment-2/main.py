import os

import numpy as np
import tensorflow as tf
import yaml

#sys.path.append("/Users/patrikkj/git/patrikkj/it3105/")
os.chdir("/Users/patrikkj/git/patrikkj/it3105")
from agents import (HumanHexAgent, HumanHexAgentV2, MCTSAgent, NaiveMCTSAgent,
                    RandomAgent)
from environment_loop import EnvironmentLoop
from envs.hex import HexEnvironment
from oht.BasicClientActor import BasicClientActor
from tournament import Tournament
from utils import debug

tf.random.set_seed(0)
np.random.seed(0)


# Load configurations
with open("./assignment-2/config.yml") as f:
   config = yaml.load(f, Loader=yaml.FullLoader)

# with open("./assignment-2/config_topp.yml") as f:
#    config = yaml.load(f, Loader=yaml.FullLoader)

#@debug
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
    if False:
        with HexEnvironment(**config["hex_params"]) as env:
            agent_2 = MCTSAgent.from_checkpoint(
                env, 
                config["export_dir"], 
                name="mctsagent__2021_04_18__17_08_30", 
                episode=1000)
            
            #agent_1 = MCTSAgent.from_config(env, config).learn()
            agent_1 = HumanHexAgentV2(env)
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
    
    # Run the OHT
    if True:
        def agent_factory(starting_player):
            env = HexEnvironment(starting_player=starting_player, **config["hex_params"])
            agent = MCTSAgent.from_checkpoint(
                    env, 
                    config["export_dir"], 
                    name="oht_v2_1000", 
                    episode=1000)
            #agent = NaiveMCTSAgent(env, n_simulations=30_000)
            return agent
        bsa = BasicClientActor(agent_factory, verbose=True)
        bsa.connect_to_server()
    
if __name__ == "__main__":
    main()
