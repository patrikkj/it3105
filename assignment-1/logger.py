import tensorflow as tf


step_logs = []
def step_logger(agent, episode, step=0):
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
def episode_logger(agent, episode):
    series = {}
    series["episode"] = episode
    series["board"] = agent.env.board.copy()
    series["n_pegs_left"] = agent.env.get_pegs_left()
    episode_logs.append(series)

def episode_reporter_wrapper(freq=50):
    def episode_reporter(agent, episode, freq=50):
        if episode % freq == 0:
            latest = episode_logs[-freq:]
            n_episodes = len(latest)
            n_pegs = [series["n_pegs_left"] for series in latest]
            n_wins = sum(n == 1 for n in n_pegs)

            ep_info = f"Episode {(episode + 1) - n_episodes} -  {episode}"
            wins_info = f"wins: {n_wins}/{n_episodes} ({(n_wins/n_episodes)*100:.2f}%)"
            epsilon_info = f"epsilon: {agent.actor._current_epsilon if agent._training else '0 (training=False)'}"
            table_info = f"𝛑-table size: {len(agent.actor.policy)}"
            tf.print(f"{ep_info:22}  {wins_info:24}  {epsilon_info:30}  {table_info}")
    return episode_reporter
