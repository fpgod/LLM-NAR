import sys
import os

sys.path.append(r"configs")
sys.path.append(r"models")
import yaml
import argparse
import random
import numpy as np
from tqdm import tqdm
import wandb
import time
import json

import torch
from torch import nn
from torch import optim

from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config8.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as config_path:
        config = yaml.load(config_path, Loader=yaml.FullLoader)

    # config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config["device"] = torch.device("cpu")

    llm_type = config["llm_type"]
    if llm_type == "llama":
        from llm_mapf.ollama_runnerv1 import LLMRunner
    if llm_type == 'deepseek':
        from llm_mapf.deepseekapi_runnerv1  import LLMRunner
        model_name = 'deepseek-chat'

    result_path = rf"results/{llm_type}"
    map_name = rf"{config['map_shape'][1]}_{config['num_agents']}_{config['obstacles']}"

    vali_file = rf"{result_path}/{map_name}"

    if not os.path.exists(vali_file):
        os.makedirs(vali_file)

    with open(rf"{vali_file}/config.yaml", "w") as config_path:
        yaml.dump(config, config_path)


    MAX_STEPS = config["max_steps"]
    TEST_EPIs = 5
    iter = 5

    models_list = [1]


    for i, model_path in enumerate(models_list):

        print(f"\n\n----- start Validation {model_path}-----")

        vali_success_rate = []
        vali_flow_time = []
        vali_losses = []
        ts = []

        vali_dir_record_base = rf"{vali_file}/record"

        if not os.path.exists(vali_dir_record_base):
            os.makedirs(vali_dir_record_base)

        existing_folders = []
        for d in os.listdir(vali_dir_record_base):
            folder_path = os.path.join(vali_dir_record_base, d)
            if os.path.isdir(folder_path) and d.isdigit():
                metrics_file = os.path.join(folder_path, "validation_metrics.json")
                if os.path.exists(metrics_file):
                    existing_folders.append(int(d))

        start_episode = max(existing_folders + [-1]) + 1
       

        for episode in range(start_episode, start_episode + TEST_EPIs):
            vali_dir_record = os.path.join(vali_dir_record_base, str(episode))
            if not os.path.exists(vali_dir_record):
                os.makedirs(vali_dir_record)


            goals = create_goals(config["board_size"], config["num_agents"])
            start = create_goals(config["board_size"], config["num_agents"])
            obstacles = create_obstacles(config["board_size"], config["obstacles"])

            env = GraphEnv(config, goal=goals, obstacles=obstacles, starting_positions=start)
            emb = env.getEmbedding()
            obs = env.reset()

            
            np.save(rf"{vali_dir_record}/goals.npy", goals)
            np.save(rf"{vali_dir_record}/obstacles.npy", obstacles)
            np.save(rf"{vali_dir_record}/starts.npy", start)


            llm_runner = LLMRunner(filepath=rf"{vali_dir_record}", exp_name="1")


            vali_action_predicts = []
            vali_action_llms = []

            action_llm = []

            done = False
            time.sleep(3)

            total_steps_per_agent = {} 
            total_steps = 0  

            total_steps_per_agent = {}  
            agent_done = [False] * env.nb_agents  

            for i in tqdm(range(MAX_STEPS)):
                llm_0_flag = 0
                
                if len(vali_action_llms) >= 2:
                    last_two = vali_action_llms[-2:]
                    if all(all(x == 0 for x in sublist) for sublist in last_two):
                        llm_0_flag = 1
                        print(i, "llm_0_flag")

                if i % iter == 0 or llm_0_flag:
                    print("reset messages")
                    gpt_output = llm_runner.start_run()

                gpt_output = llm_runner.reload_env_run(env) 
                action_llm = llm_runner._get_actions_from_response(gpt_output, env)
                vali_action_llms.append(action_llm)
                
                action_exu = action_llm

                obs, reward, done, info = env.step(action_exu, emb)

                for agent_id in range(env.nb_agents):
                    if not agent_done[agent_id]:
                        total_steps_per_agent[agent_id] = total_steps_per_agent.get(agent_id, 0) + 1
                
                agents_positions = np.array([env.positionX, env.positionY]).T
                for agent_id, pos in enumerate(agents_positions):
                    if np.array_equal(pos, env.goal[agent_id]):
                        if agent_id not in total_steps_per_agent:
                            total_steps_per_agent[agent_id] = i 
                            agent_done[agent_id] = True  

                total_steps += sum(not idone for idone in agent_done)

                if done:
                    print("done")
                    break

            
            np.save(rf"{vali_dir_record}/action_predicts.npy", np.array(vali_action_predicts).T)
            np.save(rf"{vali_dir_record}/action_llms.npy", np.array(vali_action_llms).T)

            metrics = env.computeMetrics()
            vali_success_rate.append(metrics[0])
            vali_flow_time.append(metrics[1])
            ts.append(total_steps)

            np.save(rf"{vali_dir_record}/success_rate.npy", metrics[0])
            np.save(rf"{vali_dir_record}/flow_time.npy", metrics[1])
            np.save(rf"{vali_dir_record}/total_steps.npy", total_steps)

            data = {
                "success_rate": metrics[0],  
                "success_rate%": metrics[0]/config['num_agents'],  
                "flow_time": metrics[1],
                "total_steps": total_steps
            }

            json_file_path = rf"{vali_dir_record}/validation_metrics.json"

            with open(json_file_path, 'w') as json_file:
                json.dump(data, json_file)

            print(f"Success rate: {metrics[0]}")
            

        vali_aver_success_rate0 = np.mean(vali_success_rate)
        vali_aver_flow_time0 = np.mean(vali_flow_time)
        total_steps1 = np.mean(ts)


        print(f"\n\noverall:\n")
        print(map_name)
        print(f"Success rate: {vali_aver_success_rate0}")
        print(f"Flow time: {vali_aver_flow_time0}")

        np.save(rf"{vali_file}/success_rate.npy", vali_success_rate)
        np.save(rf"{vali_file}/flow_time.npy", vali_flow_time)
        np.save(rf"{vali_file}/total_steps.npy", total_steps1)


        data = {
                "success_rate": vali_aver_success_rate0,  
                "success_rate%": vali_aver_success_rate0/config['num_agents'],  
                "flow_time": vali_aver_flow_time0,
                "total_steps": total_steps1
        }

        json_file_path = rf"{vali_file}/validation_metrics.json"
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file)

        all_success_rates = []
        all_flow_times = []
        all_total_steps = []
        
        for d in os.listdir(vali_dir_record_base):
            folder_path = os.path.join(vali_dir_record_base, d)
            metrics_file = os.path.join(folder_path, "validation_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    all_success_rates.append(metrics["success_rate"])
                    all_flow_times.append(metrics["flow_time"])
                    all_total_steps.append(metrics["total_steps"])
        
        all_success_rates.extend(vali_success_rate)
        all_flow_times.extend(vali_flow_time)
        all_total_steps.extend(ts)
        
        overall_success_rate = np.mean(all_success_rates)
        overall_flow_time = np.mean(all_flow_times)
        overall_total_steps = np.mean(all_total_steps)
        
        np.save(rf"{vali_file}/success_rate.npy", all_success_rates)
        np.save(rf"{vali_file}/flow_time.npy", all_flow_times)
        np.save(rf"{vali_file}/total_steps.npy", all_total_steps)

        data = {
            "success_rate": overall_success_rate,  
            "success_rate%": overall_success_rate/config['num_agents'],  
            "flow_time": overall_flow_time,
            "total_steps": overall_total_steps,
            "num_episodes": len(all_success_rates)  
        }

        with open(rf"{vali_file}/overall_validation_metrics.json", 'w') as json_file:
            json.dump(data, json_file)

        print(f"\n\nOverall results (including previous runs):")
        print(f"Total episodes: {len(all_success_rates)}")
        print(f"Average success rate: {overall_success_rate}")
        print(f"Average success rate%: {overall_success_rate/config['num_agents']}")
        print(f"Average flow time: {overall_flow_time}")
        print(f"Average total steps: {overall_total_steps}")


