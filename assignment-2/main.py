import yaml

from agents import (HumanHexAgent, HumanHexAgentV2, MCTSAgent, NaiveMCTSAgent,
                    RandomAgent)
from environment_loop import EnvironmentLoop
from tournament import Tournament
from envs.hex import HexEnvironment

# Load configuration
CONFIG_PATH = "./assignment-2/config.yml"
with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def main():
    with HexEnvironment(**config["hex_params"]) as env:
        #agent_1 = MCTSAgent.from_config(env, config)
        agent_1 = NaiveMCTSAgent(env, n_simulations=50)
        #agent_1 = RandomAgent(env)
        agent_2 = RandomAgent(env)

        #with EnvironmentLoop(env, agent_1, agent_2, framerate=20) as loop:
        #    loop.train_agents()
        #    loop.play_game()

        with Tournament(env, [agent_1, agent_2], num_series=200) as tournament:
            tournament.play_tournament()


if __name__ == "__main__":
    main()
