from dataset_gen import create_solutions
from trayectory_parser import parse_traject
from record import record_env


if __name__ == "__main__":
    cases = 500
    config = {
        "num_agents": 10,
        "map_shape": [20, 20],
        "nb_agents": 10,
        "nb_obstacles": 0,
        "sensor_range": 6,
        "board_size": [20, 20],
        "max_time": 100,
        "min_time": 0,  # min time the tray should go from start to goal
        "path": rf"dataset/20_10_0/",
    }

    for path in [config["path"]]:
        print("1 create_solutions")
        create_solutions(path, cases, config)
        print("\n2 parse_traject")
        parse_traject(path)
        print("\n3 record_env")
        record_env(path, config)
