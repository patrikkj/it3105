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
    if episode == 5:
        series["td_error"] = agent._error
    step_logs.append(series)

episode_logs = []
def episode_logger(agent, episode, final=False):
    series = {}
    series["episode"] = episode
    series["board"] = agent.env.board.copy()
    series["n_pegs_left"] = agent.env.get_pegs_left()
    episode_logs.append(series)

last_reported = 0
reporter_vars = {
    "last_reported": 0
}
def episode_reporter(agent, episode, final=False, freq=50):
    n_episodes = episode - reporter_vars["last_reported"]
    if (n_episodes >= freq) or final:
        latest = episode_logs[-n_episodes:]
        print(latest)
        n_pegs = [series["n_pegs_left"] for series in latest]
        n_wins = sum(n == 1 for n in n_pegs)

        ep_info = f"Episode {(episode + 1) - n_episodes} -  {episode}"
        wins_info = f"wins: {n_wins}/{n_episodes} ({(n_wins/n_episodes)*100:.2f}%)"
        epsilon_info = f"epsilon: {agent.actor._current_epsilon if agent._training else '0 (training=False)'}"
        table_info = f"𝛑-table size: {len(agent.actor.policy)}"
        tf.print(f"{ep_info:22}  {wins_info:24}  {epsilon_info:30}  {table_info}")
        reporter_vars["last_reported"] = episode
