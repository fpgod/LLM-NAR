import random
import json
import numpy as np
import yaml

action_list = np.array([[0, 0],[-1, 0],[1, 0],[0, -1],[0, 1]], dtype=int)



def map_partition(map):

    """Partitioning map into сomponents."""
    empty_list = np.argwhere(map == 0).tolist()
    empty_pos = set([tuple(pos) for pos in empty_list])

    if not empty_pos:
        raise RuntimeError("There are no empty positions found")

    partition_list = []
    while empty_pos:
        start_pos = empty_pos.pop()
        open_list = [start_pos]
        close_list = []

        while open_list:
            x, y = open_list.pop(0)
            for dx, dy in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ):
                pos = x + dx, y + dy
                if pos in empty_pos:
                    empty_pos.remove(pos)
                    open_list.append(pos)

            close_list.append((x, y))

        if len(close_list) >= 2: 
            partition_list.append(close_list)

    return partition_list

class Env:
    def __init__(
            self, 
            map_size=8, 
            obstacle_density=0, 
            num_agents=2,
            env_data=None):

        self.map_size = map_size
        self.obstacle_density = obstacle_density
        self.num_agents = num_agents
        self.num_obstacles = int(map_size * map_size * obstacle_density)

        assert self.num_obstacles <= map_size * map_size, "too many obstacles"
        assert num_agents <= map_size * map_size - self.num_obstacles, "too many agents"

        self.env_data = self._load_env_from_file(env_data)

        self._generate() # self.map, self.agents_pos, self.goals_pos, self.char_map, self.obstacles

        self.steps = 0

    def _load_env_from_file(self, file_path):
        with open(file_path, 'r') as file:
            env_data = yaml.safe_load(file)
            # print(env_data)
        return env_data

    def _generate(self):
        if self.env_data:
            self.map_size = self.env_data['map']['dimensions'][0]
            self.map = np.zeros((self.map_size, self.map_size), dtype=int)
            self.agents_pos = np.array([agent['start'] for agent in self.env_data['agents']], dtype=int)
            self.goals_pos = np.array([agent['goal'] for agent in self.env_data['agents']], dtype=int)
            self.num_agents = len(self.agents_pos)
            print(self.env_data['map']['obstacles'])
            if self.env_data['map']['obstacles']:
                self.obstacles = np.array(self.env_data['map']['obstacles'], dtype=int)
                for obstacle in self.obstacles:
                    self.map[obstacle[0], obstacle[1]] = 1
            else:
                self.obstacles = {(i, j) for i, j in np.argwhere(self.map == 1)}
            
            self._generate_char_map()
        else:
            self.map = np.zeros((self.map_size, self.map_size), dtype=int)
            all_positions = list(np.ndindex(self.map.shape))
            selected_positions = random.sample(all_positions, self.num_obstacles)
            for position in selected_positions:
                self.map[position] = 1

            # print(self.map)

            partition_list = map_partition(self.map)
            self._part = partition_list

            while len(partition_list) == 0:
                self.map = np.zeros((self.map_size, self.map_size), dtype=int)
                all_positions = list(np.ndindex(self.map.shape))
                selected_positions = random.sample(all_positions, self.num_obstacles)
                for position in selected_positions:
                    self.map[position] = 1

                # print(self.map)

                partition_list = map_partition(self.map)

            self.agents_pos = np.empty((self.num_agents, 2), dtype=int)
            self.goals_pos = np.empty((self.num_agents, 2), dtype=int)

            pos_num = sum([len(partition) for partition in partition_list])

            # loop to assign agent original position and goal position for each agent
            for i in range(self.num_agents):
                pos_idx = random.randint(0, pos_num - 1)
                partition_idx = 0
                for partition in partition_list:
                    if pos_idx >= len(partition):
                        pos_idx -= len(partition)
                        partition_idx += 1
                    else:
                        break

                pos = random.choice(partition_list[partition_idx])
                partition_list[partition_idx].remove(pos)
                self.agents_pos[i] = np.asarray(pos, dtype=int)

                pos = random.choice(partition_list[partition_idx])
                partition_list[partition_idx].remove(pos)
                self.goals_pos[i] = np.asarray(pos, dtype=int)

                partition_list = [
                    partition for partition in partition_list if len(partition) >= 2
                ]
                pos_num = sum([len(partition) for partition in partition_list])

            
            self.obstacles = {(i, j) for i, j in np.argwhere(self.map == 1)}
            self._generate_char_map()

    def _generate_char_map(self):
        self.char_map = np.full(self.map.shape, '.', dtype=str)
        self.char_map[self.map == 1] = '@'
        # print(self.char_map)

    def step(self, actions, next_pos_extract):
        """
        actions:
            list of indices
                0 stay
                1 up
                2 down
                3 left
                4 right
            action_list = np.array([[0, 0],[-1, 0],[1, 0],[0, -1],[0, 1]], dtype=np.int)

        """

        assert (
            len(actions) == self.num_agents
        ), "Error: only {} actions as input while {} agents in environment. Exiting env.step.".format(
            len(actions), self.num_agents
        )
        assert all(
            [action_idx < 5 and action_idx >= 0 for action_idx in actions]
        ), "Error: action index out of range. Exiting env.step."

        next_pos = np.copy(self.agents_pos)

        for agent_id in range(self.num_agents):
            next_pos[agent_id] += action_list[actions[agent_id]]


        # if not np.array_equal(next_pos, next_pos_extract):
        #     print("Mismatch found between next_pos and next_pos_extract")
        #     print("next_pos:")
        #     print(next_pos)
        #     print("next_pos_extract:")
        #     print(next_pos_extract)
        #     diff = next_pos != next_pos_extract
        #     print("Differences at positions:")
        #     print(np.argwhere(diff))
        # else:
        #     # print("next_pos and next_pos_extract are consistent")
        #     pass
        assert np.array_equal(next_pos, next_pos_extract), "Error: next_pos and next_pos_extract do not match. Exiting env.step."



        obstacle_collisions = []   
        agent_collisions = []      
        out_of_bounds_collisions = [] 
        any_collision = False     
        done = False

        for idx, pos in enumerate(next_pos):
            if pos[0] < 0 or pos[0] >= self.map.shape[0] or pos[1] < 0 or pos[1] >= self.map.shape[1]:
                out_of_bounds_collisions.append(idx) 
                continue

            # print(self.map)
            if self.map[pos[0], pos[1]] == 1:  
                # print(idx)
                obstacle_collisions.append(idx)  
                any_collision = True

        seen_positions = {}
        for idx, pos in enumerate(next_pos):
            if tuple(pos) in seen_positions:
                agent_collisions.append((seen_positions[tuple(pos)], idx))
                any_collision = True
            seen_positions[tuple(pos)] = idx

        self.agents_pos = np.copy(next_pos)

        if all(np.array_equal(agent_pos, goal_pos) for agent_pos, goal_pos in zip(self.agents_pos, self.goals_pos)):
            done = True  

        self.steps += 1

        conflict_info = (obstacle_collisions, agent_collisions, out_of_bounds_collisions, any_collision)

        # print("env.step")
        # print(self.agents_pos)

    
        all_valid_moves, all_descriptive_moves = self._get_all_valid_moves()

        valid_moves_info = (all_valid_moves, all_descriptive_moves)

        rewards = None

        return self.observe(), rewards, done, conflict_info, valid_moves_info
    
    def _get_all_valid_moves(self):
        all_valid_moves = []
        all_descriptive_moves = []

        for agent_idx, agent_pos in enumerate(self.agents_pos):
            valid_moves, descriptive_moves = self._get_valid_moves(agent_pos)
            all_valid_moves.append((agent_idx + 1, valid_moves))
            all_descriptive_moves.append((agent_idx + 1, descriptive_moves))

        return all_valid_moves, all_descriptive_moves

    def _get_valid_moves(self, agent_pos):
        x, y = agent_pos
        
        agent_positions = {tuple(pos) for pos in self.agents_pos if not np.array_equal(pos, agent_pos)}

        valid_moves = []
        descriptive_moves = []

        if x - 1 >= 0 and (x - 1, y) not in self.obstacles and (x - 1, y) not in agent_positions:
            valid_moves.append(0)
            descriptive_moves.append(f'move up to ({x - 1}, {y})')

        if x + 1 < self.map_size and (x + 1, y) not in self.obstacles and (x + 1, y) not in agent_positions:
            valid_moves.append(1)
            descriptive_moves.append(f'move down to ({x + 1}, {y})')

        if y - 1 >= 0 and (x, y - 1) not in self.obstacles and (x, y - 1) not in agent_positions:
            valid_moves.append(2)
            descriptive_moves.append(f'move left to ({x}, {y - 1})')

        if y + 1 < self.map_size and (x, y + 1) not in self.obstacles and (x, y + 1) not in agent_positions:
            valid_moves.append(3)
            descriptive_moves.append(f'move right to ({x}, {y + 1})')

        valid_moves.append(4)
        descriptive_moves.append(f'stay at ({x}, {y})')

        return valid_moves, descriptive_moves

    def observe(self):
        # This function should return the current state of the environment
        # Typically this would be formatted in a way suitable for an RL agent
        # print(
        #     "map", self.map,
        #     "agents_pos", self.agents_pos,
        #     "goals_pos", self.goals_pos,
        #     "obstracles", self.obstacles,
        #     "steps", self.steps
        # )
        obs = None
        return obs, np.copy(self.agents_pos)
    
    def load(self):
        pass

    def render(self):
        for row in self.map:
            print(" ".join(row))
        print("Agents and Goals:")
        for agent, goal in zip(self.agents_pos, self.goals_pos):
            print(f"Agent: {agent} -> Goal: {goal}")
    
    def reset(self):
        pass

# env = Env(map_size=4, obstacle_density=0.2, num_agents=3)
# env.observe()

# env_data = 'input.yaml'
# env = Env(env_data=env_data)
# env.observe()