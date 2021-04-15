import yaml

from agents import (HumanHexAgent, HumanHexAgentV2, MCTSAgent, NaiveMCTSAgent,
                    RandomAgent)
from environment_loop import EnvironmentLoop
from envs.hex import HexEnvironment
from tournament import Tournament
from utils import debug

# Load configuration
CONFIG_PATH = "./assignment-2/config.yml"
with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def main():
    with HexEnvironment(**config["hex_params"]) as env:
        # Agents
        random = RandomAgent(env)
        human = HumanHexAgentV2(env)                   
        mcts = MCTSAgent.from_config(env, config).learn()
        #mcts_ckpt = MCTSAgent.from_checkpoint(env, config["export_dir"], episode=0)

        # Custom
        if False:
            ...

        # Deliverable 1
        if False:
            ...


        # Deliverable 2
        if False:
            agent_1 = MCTSAgent.from_config(env, config).learn()
            agent_2 = HumanHexAgentV2(env)
            EnvironmentLoop(env, agent_1, agent_2, framerate=10).play_game()

        # Deliverable 3
        if False:
            agents = MCTSAgent.from_agent_directory(env=env, export_dir=config["export_dir"])
            Tournament(env, agents, **config["topp_params"]).play_tournament()


        # Deliverable 4
        if False:
            ...

        # Deliverable 2
        # Agent 1
        #agent_1 = MCTSAgent.from_config(env, config)
        # agent_1 = mcts_ckpt
        # agent_1 = random
        
        # Agent 2
        # agent_2 = HumanHexAgentV2(env)
        
        # Play visuzliation game against random agent
        #if True:
        #    EnvironmentLoop(env, agent_1, agent_2, framerate=10).play_game()
        # if False:
        #     EnvironmentLoop(env, agent_1, agent_2, framerate=20).train_agents().play_game()
        # if False:
        #     Tournament(env, [agent_1, agent_2], num_series=10).play_tournament()
        if False:
            agents = MCTSAgent.from_agent_directory(env=env, export_dir=config["export_dir"])
            Tournament(env, agents, **config["topp_params"]).play_tournament()


if __name__ == "__main__":
    main()
