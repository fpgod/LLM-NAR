import json
import requests
import random
import os
import re
import numpy as np
import http.client
import json

from openai import OpenAI

from environment import Env



class LLMRunner:
    def __init__(
            self,
            filepath,
            # exp_name, 
            # env,
            # num_iterations=5
            ):
        
        self.model = "deepseek"
        self.messages = []

        self.filepath = filepath
        self.max_steps = 1000

        self.steps = 0

        self.role_prompt_path = "llm_mapf/role_prompt.txt"

        with open(self.role_prompt_path, 'r', encoding='utf-8') as file:
            self.role_prompt = file.read()

        self.filename = f"{self.filepath}/llm_record.json"
        
        print("**llm record file:", self.filename)



        if not os.path.exists(f"{self.filepath}/"):
            os.makedirs(f"{self.filepath}/")

        
        with open(self.filename, 'w') as file:
            json.dump([], file, ensure_ascii=False, indent=4)  # Create an empty JSON array


    def chat(self, role, prompt):
        client = OpenAI(api_key="your api key", base_url="https://api.deepseek.com")

        user_input = prompt
        if not user_input:
            exit()


        message_to0 = {"role": role, "content": user_input}

        if role == 'system':
            message_to = [
                {"role": role, "content": user_input},
            ]
        else:
            message_to=[
                {"role": "system", "content": self.role_prompt},
                {"role": "user", "content": user_input},
            ]

        self.messages.append(message_to0)


        new_message = []
        new_message.append(message_to0) 

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=message_to,
            stream=False
        )

        if response.choices[0].message:
            messagefromapi = response.choices[0].message
            if messagefromapi.content:
                content = messagefromapi.content
                role = messagefromapi.role
                messagefromapijson = {"role": role, "content": content} 
                # print(messagefromapijson)
                self.messages.append(messagefromapijson)

                new_message.append(messagefromapijson)
           
                try:
                    with open(self.filename, 'r') as file:
                        try:
                            existing_data = json.load(file)
                        except json.JSONDecodeError:
                            existing_data = []
                except FileNotFoundError:
                    existing_data = []

                if isinstance(existing_data, list):
                    existing_data.extend(new_message)
                else:
                    existing_data = new_message


                with open(self.filename, 'w') as file:
                    file.write(json.dumps(existing_data, ensure_ascii=False, indent=4))


                return messagefromapijson, content
            
            return None, None
            
        return None, None
    

    def start_run(self):
       
        self.messages = []
        response = None
        role = 'system'
        _, response = self.chat(role, self.role_prompt)
        return response
    
    def reload_env_run(self, env):
        task_prompt = self._generate_task_prompt(env)
       
        response = None
        role = 'user'
        _, response = self.chat(role, task_prompt)
        return response
        
    def _get_actions_from_response(self, response, env):
        next_positions, actions = self._get_actions(response, env)
        actions_llm = self._determine_actions(next_positions, env)

       
        if len(actions_llm) != env.nb_agents:
            padding_length = env.nb_agents - len(actions_llm)
            
            padding = np.zeros(padding_length, dtype=actions_llm.dtype)
          
            actions_llm = np.concatenate((actions_llm, padding))
            print("     not long enough:", actions_llm)
        return actions_llm
    
    def _determine_actions(self, next_positions, env):
      
        action_ids = []

        agents_pos = np.column_stack((env.positionX, env.positionY))
        
    
        for idx, current_pos in enumerate(agents_pos, start=1):
            
        
            next_pos = next_positions.get(idx)

       
            if next_pos:
             
                delta_x = next_pos[0] - current_pos[0]
                delta_y = next_pos[1] - current_pos[1]
                
        
                found_action = False
                for action_id, (dx, dy) in env.action_list.items():
                    if (delta_x, delta_y) == (dx, dy):
                        action_ids.append(action_id)
                        found_action = True
                        break

                if not found_action:
                    action_ids.append(0) 
            else:
              
                action_ids.append(0)
            
        return action_ids
    
    def checker_run(self, conflict_info, valid_actions_info):
        checker_prompt = self._generate_checker_prompt(conflict_info, valid_actions_info)
        print(checker_prompt)
        _, response = self.chat(checker_prompt)
        return response

    def run(self):
        response  = self.start_run()
        done = False
      
        while not done:
            print('step: ', self.env.steps+1)
            response = self.reload_env_run()

            actions_llm= self._get_actions_from_response(response)

            (_, next_pos), _, done, conflict_info, valid_actions_info = self.env.step(actions_llm)  # checker在其中实现

            print("actions")
            print(actions_llm)
            print()


            if done:
                print("task success")
                with open(self.filename, 'a') as file:
                    file.write(f'\n\nsuccess\n{self.env.steps}\n')
                break
            
   
            response = self.checker_run(conflict_info, valid_actions_info)
            

            

    def llm_run(self):
        

        done = False
        while not done:
            # self.env.observe()

            print('step: ', self.env.steps+1)
            if self.env.steps % 10 == 0:
                task_prompt = self._generate_task_prompt()
                _, response = self.chat(task_prompt)

       
            actions, next_pos_ext = self._get_actions(response)

            if len(actions) != self.env.num_agents:
                padding_length = self.env.num_agents - len(actions)

                padding = np.zeros(padding_length, dtype=actions.dtype)

                actions = np.concatenate((actions, padding))
         
    
      
            
            (_, next_pos), _, done, conflict_info, valid_actions_info = self.env.step(actions, next_pos_ext) 

      

            if done:
                print("task success")
                with open(self.filename, 'a') as file:
                    file.write(f'\n\nsuccess\n{self.env.steps}\n')
                break
            
            print(conflict_info)
            print(valid_actions_info)

            checker_prompt = self._generate_checker_prompt(conflict_info, valid_actions_info)
            print(checker_prompt)
            _, response = self.chat(checker_prompt)



    def _generate_task_prompt(self, env):
        ''' graph env '''
        
        map_prompt = ""


        agents_pos = np.column_stack((env.positionX, env.positionY))
        agents_positions = agents_pos
        goals_positions = env.goal


        for i, (agent_pos, goal_pos) in enumerate(zip(agents_positions, goals_positions)):
            map_prompt += f"Agent {i + 1} is at ({agent_pos[0]}, {agent_pos[1]}), wants to go to ({goal_pos[0]}, {goal_pos[1]}).\n"


        grid = env.char_map


        map_prompt += "The map is as follows, where '@' denotes a cell with an obstacle that an agent cannot pass, "
        map_prompt += "and '.' denotes an empty cell that an agent can pass.\n"
        map_prompt += f'The lower-left cell is (0,0) and the lower-right cell is (0,{grid.shape[1] - 1}):\n'

     
        for row in grid:
            map_prompt += "".join(row) + '\n'

   
        map_prompt += "the coordinates of the obstacles: "
        if env.obstacles is not None:
          
            for i in range(env.obstacles.shape[0]):
                map_prompt += f"({env.obstacles[i, 0]},{env.obstacles[i, 1]}) "
        map_prompt += '\n'
   

    
        map_prompt += "the coordinates of the agents: "
        for agent_pos in agents_positions:
            map_prompt += f"({agent_pos[0]},{agent_pos[1]}) "
        map_prompt += '\n'

 

        return map_prompt




    def _generate_checker_prompt(self, conflict_info, valid_actions_info):
        (obstacle_collisions, agent_collisions, out_of_bounds_collisions, any_collision) = conflict_info
        (all_valid_moves, all_descriptive_moves) = valid_actions_info
        output = ""
        

        if any_collision:
            output += "[[Failure]] You are wrong. There were collisions in this step:\n"

            if out_of_bounds_collisions:
                for idx in out_of_bounds_collisions:
                    output += f'Agent {idx + 1} is outside the map.\n'
        

            if obstacle_collisions:
                for idx in obstacle_collisions:
                    output += f'Agent {idx + 1} is colliding with an obstacle.\n'
            

            if agent_collisions:
                for idx1, idx2 in agent_collisions:
                    output += f'Agent {idx1 + 1} is colliding with Agent {idx2 + 1}.\n'

            output += 'Please correct the current step.\n'
        

        else:
            output = "[[Success]] Well done. Keep going. In the next step:\n"

            for agent_index, moves in all_descriptive_moves:
                moves_str = ', '.join(moves)
                output +=(f"Agent {agent_index} can choose the following moves: {moves_str}.\n")
            
        
        return output
    
    def _get_actions(self, response, env):
        
        if not isinstance(response, str):
            print(response)
            current_positions = {}
            for idx in range(env.nb_agents):
                current_position = (env.positionX[idx], env.positionY[idx])
                current_positions[idx] = current_position
            no_action = np.array([0]*env.nb_agents)  
            return current_positions, no_action
        

        next_positions_dict = self._extract_next_positions(response, env)


        actions_text = self._extract_actions(response)
        encoded_actions = [self._encode_action(action) for action in actions_text]
        actions = np.array(encoded_actions)


        return next_positions_dict, actions

    def _extract_next_positions(self, text, env):
        total_agents = env.nb_agents
       
        pattern = r"Agent\s+(\d+):.*?Next Position:\s*\((\d+),\s*(\d+)\)"
    
        matches = re.finditer(pattern, text, re.DOTALL)

        agents = {}
        for match in matches:
            agent_id = int(match.group(1))
            if match.group(2) and match.group(3):
                next_position = (int(match.group(2)), int(match.group(3)))
            else:
                next_position = None
            agents[agent_id] = next_position

        return agents
    
    def _extract_actions(self, input_text):
        pattern = re.compile(r"Action: ([^\n]+)")
        actions = pattern.findall(input_text)
        return actions
    
    def _encode_action(self, action):
        if "Move right" in action:
            return 1
        elif "Move up" in action:
            return 2
        elif "Move left" in action:
            return 3
        elif "Move down" in action:
            return 4
        elif "Stay at" in action:
            return 0
        else:
            return -1
        
        
