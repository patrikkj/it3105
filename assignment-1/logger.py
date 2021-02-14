
step_logs = []
def on_step_end(agent, episode, step):
    series = {}
    series["episode"] = episode
    series["step"] = step
    series["board"] = agent.env.board.copy()
    series["n_pegs_left"] = agent.env.get_pegs_left()
    series["peg_move_direction"] = agent.env._peg_move_direction
    series["peg_start_position"] = agent.env._peg_start_position
    series["peg_end_position"] = agent.env._peg_end_position
    step_logs.append(series)

episode_logs = []
def on_episode_end(agent, episode):
    series = {}
    series["episode"] = episode
    series["board"] = agent.env.board.copy()
    series["n_pegs_left"] = agent.env.get_pegs_left()
    episode_logs.append(series)
