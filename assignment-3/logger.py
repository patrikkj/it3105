import tensorflow as tf


step_logs = []
def step_logger(agent, episode, step=0):
    series = {}
    series["episode"] = episode
    series["step"] = step
    series["x"] = agent.env.x
    series["v"] = agent.env.v
    step_logs.append(series)

episode_logs = []
def episode_logger(agent, episode, final=False):
    series = {}
    series["episode"] = episode
    series["x"] = agent.env.x
    series["v"] = agent.env.v
    series["complete"] = agent.env.is_completed()
    episode_logs.append(series)

REPORT_FREQ = 50
reporter_vars = {
    "last_reported": 0
}
def episode_reporter(agent, episode, final=False):
    n_episodes = episode - reporter_vars["last_reported"]
    if (n_episodes >= REPORT_FREQ) or final:
        latest = episode_logs[-n_episodes:]
        completes = [series["complete"] for series in latest]
        n_wins = sum(n == 1 for n in completes)

        ep_info = f"Episode {(episode + 1) - n_episodes} -  {episode}"
        wins_info = f"wins: {n_wins}/{n_episodes} ({(n_wins/n_episodes)*100:.2f}%)"
        epsilon_info = f"epsilon: {agent.actor._current_epsilon if agent._training else '0 (training=False)'}"
        table_info = f"𝛑-table size: {len(agent.actor.policy)}"
        tf.print(f"{ep_info:22}  {wins_info:24}  {epsilon_info:30} ")
        reporter_vars["last_reported"] = episode
