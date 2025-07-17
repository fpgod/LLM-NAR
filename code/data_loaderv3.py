import os
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import DataLoader


class MAPFDataLoader:
    def __init__(self, config):
        self.config = config

        train_set = CreateDataset(self.config, "train")

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )


class CreateDataset(data.Dataset):
    def __init__(self, config, mode):
        """
        Args:
            dir_path (string): Path to the directory with the cases.
            A case dir contains the states and trajectories of the agents
        """
        self.config = config[mode]
        self.dir_path = self.config["root_dir"]

        self.cases = [f for f in os.listdir(self.dir_path) if f.startswith('case_')]


        self.count = 0
        self.data = []

        for i, case in enumerate(self.cases):
            if os.path.exists(os.path.join(self.dir_path, case, "trajectory_record.npy")) and os.path.exists(
                os.path.join(self.dir_path, case, "tray_llm.npy")):
                tray_llm = np.load(os.path.join(self.dir_path, case, "tray_llm.npy")).T
                tray = np.load(os.path.join(self.dir_path, case, "trajectory_record.npy"))
                state = np.load(os.path.join(self.dir_path, case, "states.npy"))
                gso = np.load(os.path.join(self.dir_path, case, "gso.npy"))

                gso = gso[:, 0, :, :] + np.eye(self.config["nb_agents"])

                num_time_points = tray_llm.shape[1]
                for t in range(num_time_points):
                    self.data.append({
                        'tray_llm': tray_llm[:, t],
                        'tray': tray[:, t],
                        'state': state[t],
                        'gso': gso[t]
                    })
            self.count += 1

        print(f"Loaded {self.count} cases, has {len(self.data)} data")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the whole case
        state : (agents, channels, dimX, dimY),
        trayec: (agents)
        gsos: (agents, nodes, nodes)
        """
        data_point = self.data[index]
        trayec = torch.from_numpy(data_point['tray']).float()
        trayel = torch.from_numpy(data_point['tray_llm']).float()
        states = torch.from_numpy(data_point['state']).float()
        gsos = torch.from_numpy(data_point['gso']).float()

        return trayel, trayec, states, gsos
    
    def pad_array(self, array, target_length, num_agents):
        """Pad the array along the second dimension to the target length with zeros."""
        current_length = array.shape[1]
        if current_length < target_length:
            padding_length = target_length - current_length
            padding = np.zeros((array.shape[0], padding_length, array.shape[2])) if array.ndim == 3 else np.zeros((array.shape[0], padding_length))
            padded_array = np.concatenate((array, padding), axis=1)
        else:
            padded_array = array[:, :target_length]
        return padded_array


if __name__ == "__main__":
    config = {
        "batch_size": 3,
        "num_workers" :3,
        "train": {
            "root_dir": r"/dataset/...",
            "mode": "train",
            "max_time": 50,
            "nb_agents": 8,
            "min_time": 10,
            "max_time_dl": 50,
            "batch_size": 1,
        },
    }

    data_loader = MAPFDataLoader(config)
    print(data_loader.train_loader)
    # print(next(iter(data_loader.train_loader)))
    trayel, trayel = next(iter(data_loader.train_loader))
    print("Train:")
    print(f"llm batch shape: {trayel.size()}")
    print(f"cbs batch shape: {trayel.size()}")
