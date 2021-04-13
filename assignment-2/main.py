import yaml

from agents import (HumanHexAgent, HumanHexAgentV2, MCTSAgent, NaiveMCTSAgent,
                    RandomAgent)
from environment_loop import EnvironmentLoop
from tournament import Tournament
from envs.hex import HexEnvironment
from utils import debug

# Load configuration
CONFIG_PATH = "./assignment-2/config.yml"
with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def main():
    with HexEnvironment(**config["hex_params"]) as env:
        # Agent 1
        #agent_1 = RandomAgent(env)
        #agent_1 = HumanHexAgentV2(env)
        # agent_1 = NaiveMCTSAgent(env, n_simulations=400)
        #agent_1 = MCTSAgent.from_config(env, config)
        #agent_1 = MCTSAgent.from_checkpoint(
        #    env=env, 
        #    export_dir=config["export_dir"], 
        #    name="mctsagent__2021_04_13__15_53_57", 
        #    episode=200)

        # Agent 2
        #agent_2 = RandomAgent(env)
        # agent_2 = HumanHexAgentV2(env)
        #agent_2 = NaiveMCTSAgent(env, n_simulations=4_000)
        #agent_2 = MCTSAgent.from_config(env, config)
        #agent_2 = MCTSAgent.from_checkpoint(
        #    env=env, 
        #    export_dir=config["export_dir"], 
        #    name="mctsagent__2021_04_13__02_03_34", 
        #    episode=50)

        # with EnvironmentLoop(env, agent_1, agent_2, framerate=20) as loop:
            # loop.train_agents()
            # loop.play_game()

        # with EnvironmentLoop(env, agent_1, agent_2, framerate=20) as loop:
        #     loop.play_game()
        

        agents = MCTSAgent.from_agent_directory(
            env=env, 
            export_dir=config["export_dir"], 
            #name="mctsagent__2021_04_13__15_53_57"
            )
            
        with Tournament(env, agents, num_series=250) as tournament:
            tournament.play_tournament()


if __name__ == "__main__":
    main()
