

# Sketchup of refactored component structures
"""
env = HexEnrivonment()                              # Instance of 'StateManager'
    spec = EnvironmentSpec()
    grid = HexGrid()
    renderer = HexRenderer()

mcts_agent = MCTSAgent(env, actor, learner)         # Instance of 'LearningAgent' <- 'Agent'
    actor = MCTSActor(env, network)                             # Instance of 'Actor'
        network = Network()
    learner = MCTSLearner(env, actor, network, replay_buffer)   # Instance of 'Learner'
        replay_buffer = ReplayBuffer()
        root = MonteCarloNode()
        mct = MonteCarloTree(root)

random_agent = RandomAgent(env)                     # Instance of 'Agent'
human_agent = HumanAgent(env)                       # Instance of 'Agent'

environment_loop = EnvironmentLoop(
    agent_1=mcts_agent,
    agent_2=random_agemt
):
    if isinstance(agent, LearningAgent):
        agent.learn()
    ...
"""