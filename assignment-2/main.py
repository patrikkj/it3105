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
        # Agents
        random = RandomAgent(env)
        human = HumanHexAgentV2(env)
        mcts_naive = NaiveMCTSAgent(env, n_simulations=4000)
        mcts = MCTSAgent.from_config(env, config)
        #mcts_ckpt = MCTSAgent.from_checkpoint(env, config["export_dir"],
        #    name="mctsagent__2021_04_14__17_15_55", episode=200)

        # Agent 1
        agent_1 = MCTSAgent.from_config(env, config).learn()
        #agent_1 = mcts_ckpt
        
        # Agent 2
        agent_2 = HumanHexAgentV2(env)
        
        # Play visuzliation game against random agent
        if True:
            EnvironmentLoop(env, agent_1, agent_2, framerate=20).play_game()
        if False:
            EnvironmentLoop(env, agent_1, agent_2, framerate=20).train_agents().play_game()
        if False:
            Tournament(env, [agent_1, agent_2], num_series=10).play_tournament()
        if False:
            agents = MCTSAgent.from_agent_directory(env=env, export_dir=config["export_dir"])
            Tournament(env, agents, num_series=10).play_tournament()


if __name__ == "__main__":
    main()
