import sys
import os

sys.path.append(r"configs")
sys.path.append(r"models")
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import wandb
import json
import shutil
import random

import torch
from torch import nn
from torch import optim

from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from MAPF_GNN_master.data_generation.record import record_env, make_env
from environment import Env
from llm_mapf.deepseekapi_runnerv1 import LLMRunner
from flamingo_pytorch.flamingo_mapf import FlamingoMAPF
from data_loaderv3 import MAPFDataLoader 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as config_path:
        config = yaml.load(config_path, Loader=yaml.FullLoader)

    # config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config["device"] = torch.device("cpu")
    
    model_name = config["model_name"]
    map_name = rf"{config['map_shape'][1]}_{config['num_agents']}_{config['obstacles']}"
    exp_name = rf"21_{model_name}_{map_name}_lr_e{config['epochs']}_b{config['batch_size']}"

    net_type = config["net_type"]
    msg_type = config["msg_type"]
    if net_type == "gnn":
        if msg_type == "message":
            from MAPF_GNN_master.models.framework_gnn_message import Network
        else:
            from MAPF_GNN_master.models.framework_gnn import Network as GNN_network

    if net_type == "baseline":
        from MAPF_GNN_master.models.framework_baseline import Network



    result_path = rf"results/trained_models/{exp_name}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    trained_model_file = rf"{result_path}/models/"
    if not os.path.exists(trained_model_file):
        os.makedirs(trained_model_file)

    vali_file = rf"{result_path}/vali_dataset/"
    if not os.path.exists(vali_file):
        os.makedirs(vali_file)

    with open(rf"{result_path}/config.yaml", "w") as config_path:
        yaml.dump(config, config_path)

    config_data = config['dataset']


    data_loader = MAPFDataLoader(config)

    gnn_model = GNN_network(config)
    gnn_model.to(config["device"])
    gnn_model.load_state_dict(torch.load(rf"MAPF_GNN_master/trained_models/gnn_k3/model.pt"))
    gnn_model.eval()

    gnn_dim = 128
    model_dim = 256 #64
    depth = 3
    model = FlamingoMAPF(gnn_dim, model_dim, depth)

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    model.to(config["device"])


    losses_final = []
    aver_losses_final = []
    success_rate_final = []
    flow_time_final = []


    ####### Training #########
    print("----- Start Training -----")
    NUM_EPOCHS = config["epochs"]
    print("num_epochs", NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch}")

        
        model.train()
        epoch_loss = 0

        train_loader = tqdm(data_loader.train_loader, desc="Training", total=len(data_loader.train_loader))

        print("batches:", len(data_loader.train_loader))

        for i, (tray_llm, tray_cbs, states, gso) in enumerate(train_loader):

            optimizer.zero_grad()

            states = states.to(config["device"])
            gso = gso.to(config["device"])
            tray_llm = tray_llm.to(config["device"])
            tray_cbs = tray_cbs.to(config["device"])
            
            action_llm = tray_llm
            with torch.no_grad():
                action_gnn, feature_gnn = gnn_model(states, gso)
        

            transformed_feature_gnn = feature_gnn.permute(0, 2, 1).unsqueeze(2)
            transformed_feature_gnn = transformed_feature_gnn.to(config["device"])


            action_llm = action_llm.to(torch.int64).to(config["device"])
            transformed_action_llm = torch.zeros((action_llm.shape[0], action_llm.shape[1], 5), dtype=torch.float32)
            action_llm = action_llm.unsqueeze(2)
            transformed_action_llm.scatter_(2, action_llm, 1)
            transformed_action_llm = transformed_action_llm.to(config["device"])

            action_predict = model(transformed_action_llm, transformed_feature_gnn) #torch.Size([1, num_agents, 5])shape
            

            batch_loss = torch.zeros(1, requires_grad=True)
            for agent in range(tray_cbs.shape[1]):
                
                target = tray_cbs[:, agent].long()
                per_agent_loss = criterion(action_predict[:, agent, :], target)
                
                batch_loss = batch_loss + (per_agent_loss / tray_cbs.shape[1])

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss
            train_loader.set_postfix(loss=batch_loss.item())
        

        aver_batch_loss = epoch_loss.item()/len(data_loader.train_loader)

        losses_final.append(epoch_loss.item())
        aver_losses_final.append(aver_batch_loss)


        print(f"Epoch Loss: {epoch_loss.item()}")
        print(f"Average batch Loss: {aver_batch_loss}")

        scheduler.step()

        if (epoch + 1) % 25 != 0:
            torch.save(model.state_dict(), rf"{trained_model_file}/model_{epoch}.pt")

        print(f"\n\n----- start Validation epoch{epoch}-----")

        cases = [f for f in os.listdir(config_data['path']) if f.startswith('case_')]
        if len(cases) >= 2:
            vali_cases = random.sample(cases, 2)
        else:
            print("Not enough cases to select 50 unique items.")
            vali_cases = cases  

        model.eval()
        vali_success_rate = []
        vali_flow_time = []
        vali_losses = []

        for episode, casei in enumerate(vali_cases):
            vali_dir_record = rf"{vali_file}/model_{epoch}/record/{casei}"
            if not os.path.exists(vali_dir_record):
                os.makedirs(vali_dir_record)

            env = make_env(os.path.join(config_data['path'], casei), config_data)
            emb = env.getEmbedding()
            obs = env.reset()

            np.save(rf"{vali_dir_record}/goals.npy", env.goal)
            np.save(rf"{vali_dir_record}/obstacles.npy", env.obstacles)
            np.save(rf"{vali_dir_record}/starts.npy", np.column_stack((env.positionX, env.positionY)))

            source_path = os.path.join(config_data['path'], casei, 'solution.yaml')  
            target_path = os.path.join(vali_dir_record, 'solution.yaml')
            shutil.copy(source_path, target_path)


            valitray_cbs = np.load(os.path.join(config_data['path'], casei, "trajectory_record.npy"))  #[numagents, time]
            valitray_cbs = valitray_cbs.T

            llm_runner = LLMRunner(filepath=rf"{vali_dir_record}", exp_name=rf"{model_name}")

            vali_action_predicts = []
            vali_action_gnns = []
            vali_action_llms = []

            valiepisode_loss = 0

            MAX_STEPS = config["max_steps"]
            MAX_STEPS = 40

            for i in tqdm(range(MAX_STEPS)):
                if i % 10 == 0 :
                    print("reset messages")
                    gpt_output = llm_runner.start_run()

                gpt_output = llm_runner.reload_env_run(env) 
                action_llm = llm_runner._get_actions_from_response(gpt_output, env) 
                print("action_llm", action_llm)
                vali_action_llms.append(action_llm)
                
                fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(config["device"])
                gso = (
                    torch.tensor(obs["adj_matrix"])
                    .float()
                    .unsqueeze(0)
                    .to(config["device"])
                )
                with torch.no_grad():
                    action_gnn, feature_gnn = gnn_model(fov, gso)

                action_gnn = action_gnn.cpu().detach().squeeze(0).numpy()
                action_gnn = np.argmax(action_gnn, axis=1)

                vali_action_gnns.append(action_gnn)

                transformed_feature_gnn = feature_gnn.permute(0, 2, 1).unsqueeze(2)
                action_tensor = torch.tensor(action_llm, dtype=torch.int64)
                transformed_action_llm = torch.zeros((1, env.nb_agents, 5), dtype=torch.float32)
                
                action_tensor = action_tensor.unsqueeze(0).unsqueeze(2)
                transformed_action_llm.scatter_(2, action_tensor, 1)

                with torch.no_grad():
                    action_predict = model(transformed_action_llm, transformed_feature_gnn) 

                action_predict10 = action_predict.cpu().squeeze(0).numpy()
                action_predict10 = np.argmax(action_predict10, axis=1)
                vali_action_predicts.append(action_predict10)

                if i < valitray_cbs.shape[0]:
                    valistep_loss = torch.zeros(1, requires_grad=False)
                    # tray [time,agents]
                    tray_tensor = torch.from_numpy(valitray_cbs).long()
                    # print("tray_tensor",tray_tensor.shape)
                    for agent in range(valitray_cbs.shape[1]):
                        target = tray_tensor[i, agent]
                        if target.dim() == 0: 
                            target = target.unsqueeze(0)  
                        per_agent_loss = criterion(action_predict[:, agent, :], target)
                        valistep_loss = valistep_loss + (per_agent_loss / valitray_cbs.shape[1])
                    valiepisode_loss += valistep_loss

                obs, reward, done, info = env.step(action_predict10, emb)
                if done:
                    break


            np.save(rf"{vali_dir_record}/action_predicts.npy", np.array(vali_action_predicts).T)
            np.save(rf"{vali_dir_record}/action_gnns.npy", np.array(vali_action_gnns).T)
            np.save(rf"{vali_dir_record}/action_llms.npy", np.array(vali_action_llms).T)

            valiaverage_loss = valiepisode_loss.item()/valitray_cbs.shape[0]
            
            vali_losses.append(valiaverage_loss)

            metrics = env.computeMetrics()
            vali_success_rate.append(metrics[0])
            vali_flow_time.append(metrics[1])

            np.save(rf"{vali_dir_record}/success_rate.npy", metrics[0])
            np.save(rf"{vali_dir_record}/flow_time.npy", metrics[1])
            np.save(rf"{vali_dir_record}/aver_step_loss.npy", valiaverage_loss)

            data = {
                "success_rate": metrics[0],  
                "flow_time": metrics[1],
                "average_step_loss": valiaverage_loss
            }

            
            json_file_path = rf"{vali_dir_record}/validation_metrics.json"

            with open(json_file_path, 'w') as json_file:
                json.dump(data, json_file)

        vali_aver_epi_loss0 = np.mean(vali_losses)
        vali_aver_success_rate0 = np.mean(vali_success_rate)
        vali_aver_flow_time0 = np.mean(vali_flow_time)

        print(f"model_{epoch}_Success rate: {vali_aver_success_rate0}")
        print(f"model_{epoch}_Flow time: {vali_aver_flow_time0}")
        print(f"model_{epoch}_Average step loss: {vali_aver_epi_loss0}")


        np.save(rf"{vali_file}/model_{epoch}/success_rate.npy", vali_success_rate)
        np.save(rf"{vali_file}/model_{epoch}/flow_time.npy", vali_flow_time)
        np.save(rf"{vali_file}/model_{epoch}/aver_step_loss.npy", vali_losses)

        data = {
            "success_rate": vali_aver_success_rate0,  
            "flow_time": vali_aver_flow_time0,
            "average_step_loss": vali_aver_epi_loss0
        }
       
        json_file_path = rf"{vali_file}/model_{epoch}/validation_metrics.json"

        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file)


    loss = np.array(losses_final)
    aver_loss = np.array(aver_losses_final)

    np.save(rf"{result_path}/loss.npy", loss)
    np.save(rf"{result_path}/aver_loss.npy", aver_loss)

    torch.save(model.state_dict(), rf"{trained_model_file}/model_FINAL.pt")