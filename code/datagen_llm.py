import sys
import os

sys.path.append(r"configs")
sys.path.append(r"models")
import yaml
import argparse
import numpy as np
import tqdm
import wandb

from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from MAPF_GNN_master.data_generation.record import record_env, make_env
from llm_mapf.deepseekapi_runnerv1 import LLMRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as config_path:
        config = yaml.load(config_path, Loader=yaml.FullLoader)

    config_data = config['dataset']

    cases = [f for f in os.listdir(config_data['path']) if f.startswith('case_')]
    num_cases = len(cases)
    
    last_processed = -1
    for j in range(num_cases):
        if os.path.exists(os.path.join(config_data['path'], f"case_{j}", "tray_llm.npy")):
            last_processed = j
        else:
            print("finished cases:", last_processed)
            break

    for i in tqdm.tqdm(range(last_processed + 1, 100), desc="Cases Progress", leave=True):
        # get one case from dataset
        actions_llm = []

        gnn_env = make_env(os.path.join(config_data['path'], rf"case_{i}"), config_data)
        emb = gnn_env.getEmbedding()
        obs = gnn_env.reset()
        
        case_file = os.path.join(config_data['path'], rf"case_{i}/")
        gpt_runner = LLMRunner(case_file, f'case_{i}')

        tray_cbs = np.load(os.path.join(config_data['path'], rf"case_{i}/", "trajectory_record.npy"))  #[numagents, time]
        tray_cbs = tray_cbs.T
        
        done = False
        step_len = tray_cbs.shape[0]
        for step in tqdm.tqdm(range(step_len), desc="Steps Progress", leave=False):

            if step % 10 == 0 :
                gpt_output = gpt_runner.start_run()

            gpt_output = gpt_runner.reload_env_run(gnn_env) 
            action_llm = gpt_runner._get_actions_from_response(gpt_output, gnn_env)

            actions_llm.append(action_llm)

            obs, reward, done, info = gnn_env.step(tray_cbs[step], emb) 

            flag = 1
            if done:
                print("All agents reached their goal\n")
                break

        np.save(os.path.join(config_data['path'], rf"case_{i}/tray_llm.npy"), np.array(actions_llm))

