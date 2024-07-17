
# # **Project 2: Stock Portfolio Optimization - Assignment 3**
# Athanasakis Evangelos 2019030118 // Fragkogiannis Yiorgos 2019030039


# Importing libraries


import numpy as np
import tkinter as tk #loads standard python GUI libraries
import random
from tkinter import *
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm



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




#-------------------------------__________________Environments___________________-------------------------------------------------------------------------
# Generating environments


# Create the three different environments
# We are modeling this environment using 8 states in the format: {stock_currently_holding,state_of_stock_1,state_of_stock_2}

action_keep = 0     # keep the same stock
action_switch = 1   # switch to the other stock

# This environment is used for the question 1 where we need to demonstrate that the optimal
# policy is always to stay with the stock we already have invested
fee = -0.9
# r1H = 2*r2L
# in this case r1.h=0.1 // r2.H= 0.05 // r1.L = -0.02 // r2.L = 0.01
# we have used a large transaction fee so that the best policy will always be to keep using the same stock
P1 = {

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
            (9/20, 5, +0.05 + fee),    # {2,L,H}
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
             (9/20, 2, +0.1  + fee),  # {1,H,L}
             (1/20, 3, +0.1  + fee)   # {1,H,H}
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
            (1/20, 2, +0.1  + fee),  # {1,H,L}
            (9/20, 3, +0.1  + fee)   # {1,H,H}
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
             (9/20, 2, +0.1  + fee),  # {1,H,L}
             (1/20, 3, +0.1  + fee)   # {1,H,H}
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
             (1/20, 2, +0.1  + fee),  # {1,H,L}
             (9/20, 3, +0.1  + fee)   # {1,H,H}
        ]
    }

}


# This environment implements the stocks environment from the midterm
# It is used for the question 2 where we need to demonstrate that the optimal policy
# for some of the states is to switch and in some others to stay
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


# This environment implements the generic scenario of question 3 where for every stock
# ri_H,ri_L are chosen uniformly in [-0.02, 0.1] and transition probabilities pi_HL, pi_LH
# are equal to 0.1 for half the stocks and 0.5 for the other half.

# Since every stock can have two price states, the number of total states in the MDP
# we are creating will be = NumOfStoscks*2^numOfStocks


def decimal_to_binary_array(decimal, length):

    # Convert decimal to binary string (strip '0b' prefix)
    binary_string = bin(decimal)[2:]

    # Determine padding length
    padding_length = max(0, length - len(binary_string))

    # Pad binary string with leading zeros if needed
    padded_binary_string = '0' * padding_length + binary_string

    # Convert padded binary string to list of binary digits
    binary_array = [int(bit) for bit in padded_binary_string]

    return binary_array


# Function that generates the environment of N stocks dynamically, with a transaction fee
def generate_environment(N,fee):

    states_for_each_stock = 2**N
    total_states = N * states_for_each_stock
    max_state_length = N

    P = {}
    pi = []
    #Creating transition probabilities for the keep action
    #of EACH stock
    for i in range(0,N):
        if(i < N/2):
            # pi_HL = pi_LH = 0.1 | # pi_HH = pi_LL = 0.9
            row = [0.9,0.1,0.1,0.9] #[LL,LH,HL,HH]
        else:
            # pi_HL = pi_LH = 0.5 | # pi_HH = pi_LL = 0.5
            row = [0.5,0.5,0.5,0.5] #[LL,LH,HL,HH]
        pi.append(row)

    progress_bar = tqdm(range(0, total_states))
    for i in progress_bar:
        SubDictionary={}
        action_Keep = []
        action_Switch = []

        # find what stock we are reffering to
        # Stock ids start from 0
        stock = i // states_for_each_stock

        ##########################
        # We define states of L and H with binary ids
        # For example for 2 stocks this translation occurs:
        # LL -> 0,0 -> 0
        # LH -> 0,1 -> 1
        # HL -> 1,0 -> 2
        # HH -> 1,1 -> 3
        # The binary ids are then translated to decimals so that
        # we can use them in code
        ##########################

        current_state = i - stock * states_for_each_stock # find where this specific stock starts at the total_states environment
                                                          # this is necessary to calculate the transition probabilities

        # Convert decimal to binary string
        # Convert the binary string to a list of integers (0s and 1s)
        curr_state_array = decimal_to_binary_array(current_state, max_state_length)
        # We can now use the array to find if each stock is in high (1s) or low (0s) state
        # So We now know that we are at state {x,L,L,H....,H} with x the number of current stock

        #__Keep Stock ________________________________________________________________________________________________________________
        # progress_1 = tqdm(range (stock*2**N, ((stock+1)*2**N)))
        for j in range (stock*2**N, ((stock+1)*2**N)): # for every possible transition when keeping the same stock
            state_to_trans = j - stock * states_for_each_stock          # value (H or L) of all of the stocks at the state we will transition to, in decimal form (0,1,2,3...)
            trans_state_array = decimal_to_binary_array(state_to_trans, max_state_length) # convert to binary and take each bit separately (0 for L and 1 for H)

            transitionProb = 1

            for k in range(len(trans_state_array)):
                stock_state_trans = trans_state_array[k] # 0 or 1 // low or high
                stock_state_current = curr_state_array[k] # 0 or 1 // low or high

                if(stock_state_current == 0 and stock_state_trans == 0):       # Pi_LL
                    transitionProb = transitionProb * pi[stock][0]
                elif(stock_state_current == 0 and stock_state_trans == 1):     # pi_LH
                    transitionProb = transitionProb * pi[stock][1]
                elif(stock_state_current == 1 and stock_state_trans == 0):     # pi_HL
                    transitionProb = transitionProb * pi[stock][2]
                else:                                                          # pi_HH
                    transitionProb = transitionProb * pi[stock][3]

            nextState = j
            #reward = random.uniform(-0.02, 20)
            reward = random.uniform(-0.02, 0.1)
            action_Keep.append((transitionProb,nextState,reward))
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #fee = 0
        #__Switch Stock ________________________________________________________________________________________________________________
        # progress_bar = tqdm(range (0, total_states))
        for j in range (0, total_states): # for every possible transition when keeping the same stock
            trans_stock = j // states_for_each_stock

            if(trans_stock == stock):     # check if the transition stock is the same as the stock we start from
                continue                  # we have already handle this situation above so we move on


            trans_state = j - trans_stock * states_for_each_stock
            trans_state_array = decimal_to_binary_array(trans_state, max_state_length)
            transitionProb = 1

            for k in range(len(trans_state_array)):
                stock_state_trans = trans_state_array[k] # 0 or 1 // low or high
                stock_state_current = curr_state_array[k] # 0 or 1 // low or high

                if(stock_state_current == 0 and stock_state_trans == 0):       # Pi_LL
                    transitionProb = transitionProb * pi[stock][0]
                elif(stock_state_current == 0 and stock_state_trans == 1):     # pi_LH
                    transitionProb = transitionProb * pi[stock][1]
                elif(stock_state_current == 1 and stock_state_trans == 0):     # pi_HL
                    transitionProb = transitionProb * pi[stock][2]
                else:                                                          # pi_HH
                    transitionProb = transitionProb * pi[stock][3]

            nextState = j
            #reward = random.uniform(-0.02, 20) - fee
            reward = random.uniform(-0.02, 0.1) - fee
            action_Switch.append((transitionProb,nextState,reward))


        #-----------------------------------------------------------------------------------------------------------------------------------------------
        SubDictionary[action_keep] = action_Keep
        SubDictionary[action_switch] = action_Switch
        P[i]=SubDictionary



    return P


#==============================================================================================================================
#################### Q-Learning ################


# This function is used to simulate the environments response
# It gets as input the environment, the current state and the action that we have selected
# and it returns the next state and the reward
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

#===== Hyperparameters ===================
# alpha -> Learning rate
# gamma -> Discount factor
# epsilon ->  # Exploration rate
# epsilon_decay -> Decay rate for epsilon
# min_epsilon -> Minimum epsilon value
# num_episodes -> Number of episodes

def implement_Q_learning(environment, num_of_episodes, alpha, gamma):
    Q = np.zeros((len(environment),len(environment[0])))
    epsilon = 1.0               # Exploration rate0
    epsilon_decay = 0.99      # Decay rate for epsilon
    min_epsilon = 0.1           # Minimum epsilon value
    for _ in range(num_of_episodes):
        current_state = random.randint(0, len(environment)-1) # select a random starting state
        
        for _ in range(100):      # do 100 steps do get a feel for what happens in the environment
            # decide if we are going to explore or to exploit based on the epsilon value
            if random.uniform(0,1) < epsilon:
                #action = np.random.binomial(1,0.5)     # Explore by picking a random action
                action = random.choice([0,1])
            else:
                #action = np.argmax([Q[current_state][0],Q[current_state][1]]) # Exploit by picking the best action for this state
                #print(Q[current_state,:])
                # action = np.argmax(Q[current_state,:])
                action = np.argmax(Q[current_state])
                #print("Selected: ", action)
            next_state,reward = get_response(environment, current_state, action)
            
            # q_curr = Q[current_state][action]   # get the value of the current q based on the next state and the action that we chose
            # # target = reward + gamma * np.max(Q[next_state][:])
            # target = reward + gamma * np.max(Q[next_state])
            # Q[current_state][action] = (1-alpha) * q_curr + alpha * target

            Q[current_state,action] = Q[current_state,action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[current_state,action]
            )
            
            # update the current state
            current_state = next_state    
        
        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay 
            #print("epsilon: ",epsilon)      
        if epsilon <= min_epsilon:
            #print("alpha")
            alpha = 0.0001
        
    return Q


# #####################____TASK1____########################################
# print("\nFor environment 1 we get")   
# print(implement_Q_learning(P1, 1000, 0.9, 0.9))
# print("\nFor environment 2 we get") 
# Q2 = implement_Q_learning(P2, 1000000, 0.5, 0)  
# print(Q2)
# print(np.argmax(Q2,axis=1))

####################____TASK2____########################################
# Generating environment P3


# P3 = generate_environment(4, 0.03)
# #print(P3)
# print("\nFor environment 3 we get")   
# print(implement_Q_learning(P3, 10000, 0.1, 0.9))



####################____TASK3____########################################

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

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # output layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = self.out(x)         # Calculate output
        return x



# Class That Implements Our Deep Q-Network
class stock_market_trading_DQN():    
    # HyperParameters
    alpha = 0.001              # Learning rate
    gamma = 0              # Discount Factor
    synching_period = 10    # After this many batches we synch the target nn with the policy nn
    replay_buffer_size = 1000 # Size of replay buffer
    min_batch_size = 32      # Size of each batch
    #optimizer = optim.Adam(q_network.parameters(), lr=0.001)

    # Define Huber as our loss function
    # loss_func = nn.SmoothL1Loss()
    loss_func = nn.MSELoss()
    optimizer = None
    ACTIONS = [0,1]
    
    # Encode the input state 
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor
    
    # This function returns a random batch of size 'batch_size' from the given memory
    # def sample_mem_buffer(self, memory,batch_size):
    #     sample=[]
    #     for _ in range(batch_size):
    #         random_idx = random.randint(0,batch_size-1)
    #         sample.append(memory[random_idx])      
    #     return sample 
        
        
    # This method is responsible to train our network based on a number of 'episodes'
    def train_DQN(self, episodes,environment):
        P = environment
        num_of_states = len(P)
        num_of_actions = len(P[0])
        
        epsilon = 1 # Exploration rate
        memory_buffer = ReplayMemory(self.replay_buffer_size)
        #memory_buffer = [[] for _ in range(self.replay_buffer_size)] 
        
        #memory_buffer[i % 1000] = [0,1,2,3]
        
        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        # We create a NN with num of input nodes equal to the num of the total states 
        # The num of output layer nodes is equal to the num of the total actions
        # The hidden layer's num of nodes is equal to the num of states -> this is adjustable
        policy_dqn = DQN(in_states=num_of_states, h1_nodes=num_of_states, out_actions=num_of_actions)
        target_dqn = DQN(in_states=num_of_states, h1_nodes=num_of_states, out_actions=num_of_actions)

        # initialize the 2 networks to be the same 
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)
        print('===============================================================')
        print('===============================================================')

        # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        # self.optimizer = torch.optim.RMSprop(policy_dqn.parameters(), lr=self.alpha, alpha=0.99, 
        #                                      eps=1e-08, weight_decay=0, momentum=0, centered=False)
        
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.alpha)
        # optimizer = SGD([parameter], lr=0.1)
        
        # keep track of the reward at each round 
        reward_tracking = np.zeros(episodes)
        # List to keep track of epsilon decay
        epsilon_tracking = []
        synch_counter = 0 # which step we are on 
        
        progress_bar = tqdm(range(episodes))
        for i in progress_bar:
            current_state = random.randint(0, len(P)-1) # select a random starting state
        
            for _ in range(100):      # do 100 steps do get a feel for what happens in the environment
                # decide if we are going to explore or to exploit based on the epsilon value
                # if random.uniform(0,1) < epsilon:
                if random.random() < epsilon:
                    #action = np.random.binomial(1,0.5)     # Explore by picking a random action
                    action = random.choice([0,1])
                else:
                     # From the output layer, choose the node output (action) with the maximum value
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(current_state, num_of_states)).argmax().item()
                    
                # get the response from the environment
                next_state,reward = get_response(P, current_state, action)
                # reward_tracking[i] = reward
                
                # Store the environments response into our memory        
                # memory_buffer[step % 1000] = [current_state, action, next_state, reward]
                memory_buffer.append((current_state, action, next_state, reward)) 
            
                # update the next state
                current_state = next_state    
            
                # Increment step counter
                synch_counter += 1
            
            # Perform the optimization
            if(len(memory_buffer) > self.min_batch_size):

                #mini_batch = self.sample_mem_buffer(memory_buffer, self.min_batch_size)
                mini_batch = memory_buffer.sample(self.min_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                #epsilon = max(epsilon * 0.99, 0.1)

                # Copy policy network to target network after a certain number of steps
                ### CHECK
                # if (step % self.synching_period) == 0:
                if synch_counter > self.synching_period :
                # if (synch_counter  self.synching_period): 
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    synch_counter = 0

        # return the optimal policy
        #return policy_dqn.state_dict()
        torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")
                
    def optimize(self,mini_batch, policy_dqn, target_dqn):
        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward in mini_batch:
            # Calculate target q value 
            # We disable the gradient tracking for memory optimization
            with torch.no_grad():
                # Here we get the optimal output we SHOULD have gotten according to the target NN
                target = torch.FloatTensor(
                    # For DQNs the target NNs parameters are modified according to the equation
                    # Q[state,action] = reward + γ *max{Q[next_state]}
                    reward + self.gamma * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                )
                    
            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 

            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)

        # calculate the loss for all the batch  
        loss = self.loss_func(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model by running back-propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
    # Test function
    def test_DQN(self, episodes,environment):
        # Create FrozenLake instance
        P = environment
        num_of_states = len(P)
        num_of_actions = len(P[0])

        # Load learned policy
        policy_dqn = DQN(in_states=num_of_states, h1_nodes=num_of_states, out_actions=num_of_actions) 
 
        policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            current_state = random.randint(0, num_of_states-1)

            for _ in range(100):
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(current_state, num_of_states)).argmax().item()
                # Execute action
                current_state,reward = get_response(P, current_state, action)

        
        
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
            best_action = dqn(self.state_to_dqn_input(s, num_states)).argmax()

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end='\n')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states


 # Run the Deep - Q Learning
#if __name__ == '__main__':

dql = stock_market_trading_DQN()
# dql.train_DQN(100000,P2)
# dql.test_DQN(10,P2)
print("Generating environment:")
P3 = generate_environment(6,0.01)
dql.train_DQN(100000,P3)
dql.test_DQN(10,P3)
            
        
        
        
        
        
        
        
        
        