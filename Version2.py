import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

action_keep = 0     # keep the same stock
action_switch = 1   # switch to the other stock1


# # Define a simple model
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc = nn.Linear(10, 1)

#     def forward(self, x):
#         return self.fc(x)

# # Check for GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')
# print(torch.cuda.is_available())

# # Instantiate and move model to GPU
# model = SimpleModel().to(device)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
else:
    print("CUDA is not available. PyTorch will use the CPU.")
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = SimpleModel().to(device)




fee = -0.01
P2 = {

    # State {1,L,L}
    0:{
        action_keep: [
             (9/20, 0, -0.02),    # probability: 9/20, next_State: {1,L,L}, Reward: -0.02
             (1/20, 1, -0.02),    # {1,L,H}
             (9/20, 2, +0.1),     # {1,H,L}
             (1/20, 3, +0.1)      # {1,H,H}
        ],

        action_switch:[
            (9/20, 4, +0.01 + fee),    # {2,L,L}
            (1/20, 5, +0.05 + fee),    # {2,L,H}
            (9/20, 6, +0.01 + fee),    # {2,H,L}
            (1/20, 7, +0.05 + fee)     # {2,H,H}
        ]
    },

    # State {1,L,H}
    1:{
        action_keep: [
             (1/20, 0, -0.02),  # {1,L,L}
             (9/20, 1, -0.02),  # {1,L,H}
             (1/20, 2, +0.1 ),  # {1,H,L}
             (9/20, 3, +0.1 )   # {1,H,H}
        ],

        action_switch:[
            (1/20, 4, +0.01 + fee),    # {2,L,L}
            (9/20, 5, +0.05 + fee),    # {2,L,H}
            (1/20, 6, +0.01 + fee),    # {2,H,L}
            (9/20, 7, +0.05 + fee)     # {2,H,H}
        ]
    },

    # State {1,H,L}
    2:{
        action_keep: [
             (9/20, 0, -0.02),  # {1,L,L}
             (1/20, 1, -0.02),  # {1,L,H}
             (9/20, 2, +0.1 ),  # {1,H,L}
             (1/20, 3, +0.1 )   # {1,H,H}
        ],

        action_switch:[
            (9/20, 4, +0.01 + fee),    # {2,L,L}
            (1/20, 5, +0.05 + fee),    # {2,L,H}
            (9/20, 6, +0.01 + fee),    # {2,H,L}
            (1/20, 7, +0.05 + fee)     # {2,H,H}
        ]
    },

    # State {1,H,H}
    3:{
        action_keep: [
             (1/20, 0, -0.02),  # {1,L,L}
             (9/20, 1, -0.02),  # {1,L,H}
             (1/20, 2, +0.1 ),  # {1,H,L}
             (9/20, 3, +0.1 )   # {1,H,H}
        ],

        action_switch: [
            (1/20, 4, +0.01 + fee),    # {2,L,L}
            (9/20, 5, +0.05  + fee),    # {2,L,H}
            (1/20, 6, +0.01 + fee),    # {2,H,L}
            (9/20, 7, +0.05 + fee)     # {2,H,H}
        ]
    },

    # State {2,L,L}
    4:{
        action_keep: [
             (9/20, 4,  +0.01),    # {2,L,L}
             (1/20, 5,  +0.05),    # {2,L,H}
             (9/20, 6,  +0.01),    # {2,H,L}
             (1/20, 7,  +0.05)     # {2,H,H}
        ],

        action_switch:[
             (9/20, 0, -0.02 + fee),  # {1,L,L}
             (1/20, 1, -0.02 + fee),  # {1,L,H}
             (9/20, 2, +0.1 + fee),  # {1,H,L}
             (1/20, 3, +0.1 + fee)   # {1,H,H}
        ]
    },

    # State {2,L,H}
    5:{
        action_keep: [
             (1/20, 4, +0.01),    # {2,L,L}
             (9/20, 5, +0.05),    # {2,L,H}
             (1/20, 6, +0.01),    # {2,H,L}
             (9/20, 7, +0.05)     # {2,H,H}
        ],

        action_switch:[
            (1/20, 0, -0.02 + fee),  # {1,L,L}
            (9/20, 1, -0.02 + fee),  # {1,L,H}
            (1/20, 2, +0.1 + fee),  # {1,H,L}
            (9/20, 3, +0.1 + fee)   # {1,H,H}
        ]
    },

    # State {2,H,L}
    6:{
        action_keep: [
             (9/20, 4, +0.01),    # {2,L,L}
             (1/20, 5, +0.05),    # {2,L,H}
             (9/20, 6, +0.01),    # {2,H,L}
             (1/20, 7, +0.05)     # {2,H,H}
        ],

        action_switch:[
             (9/20, 0, -0.02 + fee),  # {1,L,L}
             (1/20, 1, -0.02 + fee),  # {1,L,H}
             (9/20, 2, +0.1 + fee),  # {1,H,L}
             (1/20, 3, +0.1 + fee)   # {1,H,H}
        ]
    },

    # State {2,H,H}
    7:{
        action_keep: [
             (1/20, 4, +0.01),    # {2,L,L}
             (9/20, 5, +0.05),    # {2,L,H}
             (1/20, 6, +0.01),    # {2,H,L}
             (9/20, 7, +0.05)     # {2,H,H}
        ],

        action_switch:[
             (1/20, 0, -0.02 + fee),  # {1,L,L}
             (9/20, 1, -0.02 + fee),  # {1,L,H}
             (1/20, 2, +0.1 + fee),  # {1,H,L}
             (9/20, 3, +0.1 + fee)   # {1,H,H}
        ]
    }

}




def get_response(environment, state, action):
    P = environment
    
    response = P[state][action] # get next states, transition probabilities and transaction rewards
                                # based on the current state and the action we want to make   

    # we use random.choices to get a random next state based on the weighted probabilities of the next states
    probabilities = []
    choices = range(len(P[state][action]))
    for i in range(len(P[state][action])): 
        probabilities.append(response[i][0])
        
     
    # because depending on the action (keep or switch) the num of actions we can take is different
    # hence, we check what the action we do is and declare the choices array accordingly
        
    # Make a random choice based on probabilities
    # k=1: Specifies that we want to make a single random choice.
    # [0] is used to extract the single element from that list
    random_choice = random.choices(choices, weights=probabilities, k=1)[0]
     
    next_state = response [random_choice][1] # get next state
    reward = response [random_choice][2]     # get reward
     
    # print("Current State: ",state)
    # print("Action: ",action)
    # print("Next State: ", next_state)
    # print("Prob: ",probabilities[random_choice])
    # print("Reward: ",reward)
    

    return next_state,reward

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = self.out(x)         # Calculate output
        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# FrozeLake Deep Q-Learning
class FrozenLakeDQL():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0         # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32            # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    ACTIONS = [0,1]     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    # Train the FrozeLake environment
    def train(self, episodes):
        # Create FrozenLake instance
        # env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)
        env = P2
        
        print(env)
        num_states = len(P2)
        num_actions = len(P2[0])
        
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
            
        for i in range(episodes):
            state = random.randint(0, len(P2)-1) 

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            # while(not terminated and not truncated):
            for _ in range(100):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = random.choice([0,1])
                    #action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state,reward  = get_response(P2, state, action)

                # Save experience into memory
                memory.append((state, action, new_state, reward)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1


            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        # Close environment
        # env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")

        # # Create new graph 
        # plt.figure(1)

        # # Plot average rewards (Y-axis) vs episodes (X-axis)
        # sum_rewards = np.zeros(episodes)
        # for x in range(episodes):
        #     sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        # plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.plot(sum_rewards)
        
        # # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        # plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.plot(epsilon_history)
        
        # # Save plots
        # plt.savefig('frozen_lake_dql.png')

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward in mini_batch:
                # Calculate target q value 
            with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes):
        # Create FrozenLake instance
        #env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human')
        
        env = P2
        
        print(env)
        num_states = len(P2)
        num_actions = len(P2[0])

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = random.randint(0, num_states-1)  # Initialize to state 0

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            for _ in range(100):  
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                state,reward = get_response(P2,state, action)

        

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' \n')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states

if __name__ == '__main__':

    frozen_lake = FrozenLakeDQL()
    is_slippery = False
    frozen_lake.train(10000)
    frozen_lake.test(10)