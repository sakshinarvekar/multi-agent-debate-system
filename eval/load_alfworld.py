"""
eval/load_alfworld.py
Loads ALFWorld environment for multi-agent debate evaluation.
"""

import os
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

ALFWORLD_CONFIG = {
    'general': {'training_method': 'dagger', 'use_cuda': False},
    'dagger': {'training': {'max_nb_steps_per_episode': 50}},
    'env': {
        'type': 'AlfredTWEnv',
        'goal_desc_human_anns_prob': 0.0,
        'domain_randomization': False,
        'rng_seed': 42,
        'task_types': [1, 2, 3, 4, 5, 6],
        'expert_type': 'handcoded',
        'reward': {'type': 'dense'}
    },
    'dataset': {
        'data_path': os.path.expanduser('~/.cache/alfworld/json_2.1.1/train'),
        'eval_id_data_path': os.path.expanduser('~/.cache/alfworld/json_2.1.1/valid_seen'),
        'eval_ood_data_path': os.path.expanduser('~/.cache/alfworld/json_2.1.1/valid_unseen'),
        'num_train_games': 100,
        'num_eval_games': 100,
    }
}


def load_alfworld_env(n=10, split='eval_in_distribution'):
    """
    Load ALFWorld environment.
    Returns env and number of games.
    split options: 'train', 'eval_in_distribution', 'eval_out_of_distribution'
    """
    config = ALFWORLD_CONFIG.copy()
    config['dataset']['num_eval_games'] = n
    config['dataset']['num_train_games'] = n

    env = AlfredTWEnv(config, train_eval=split)
    env = env.init_env(batch_size=1)
    return env


def extract_task(obs: str) -> str:
    """Extract task description from observation."""
    for line in obs.split('\n'):
        if 'your task is to:' in line.lower():
            return line.strip()
    return obs.split('\n')[0].strip()


if __name__ == "__main__":
    env = load_alfworld_env(n=5)
    obs, info = env.reset()
    print(f"Task     : {extract_task(obs[0])}")
    print(f"Obs      : {obs[0][:200]}")
    print(f"Commands : {list(info['admissible_commands'])[0][:5]}")