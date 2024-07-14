
# # **Project 2: Stock Portfolio Optimization - Assignment 3**
# Athanasakis Evangelos 2019030118 // Fragkogiannis Yiorgos 2019030039


# Importing libraries


import numpy as np
import tkinter as tk #loads standard python GUI libraries
import numpy as np
import time
import random
from tkinter import *


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


    for i in range(0, total_states):
        SubDictionary={}
        action_Keep = []
        action_Switch = []

        # find what stock we are reffering to
        # Stock ids start from 0
        stock = i // states_for_each_stock

        ##########################
        # We define states of L and H with binary ids
        # For example for 2 stocks this stranslation occurs:
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
            #reward = random.uniform(-0.02, 2)
            reward = random.uniform(-0.02, 0.1)
            action_Keep.append((transitionProb,nextState,reward))
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #fee = 0
        #__Switch Stock ________________________________________________________________________________________________________________
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
            #reward = random.uniform(-0.02, 2) - fee
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
    choices = [0,1,2,3]
    probabilities = [response[0][0], response[1][0], response[2][0], response[3][0]]
    # Make a random choice based on probabilities
    # k=1: Specifies that we want to make a single random choice.
    # [0] is used to extract the single element from that list
    random_choice = random.choices(choices, weights=probabilities, k=1)[0]
     
    next_state = response [random_choice][1] # get next state
    reward = response [random_choice][2]     # get reward
     
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
    epsilon = 1.0             # Exploration rate
    epsilon_decay = 0.99      # Decay rate for epsilon
    min_epsilon = 0.1         # Minimum epsilon value
    for ep in range(num_of_episodes):
        current_state = random.randint(0, 7) # select a random starting state
        
    for _ in range(100):      # do 100 steps do get a feel for what happens in the environment
        # decide if we are going to explore or to exploit based on the epsilon value
        if random.uniform(0,1) < epsilon:
            action = np.random.binomial(1,0.5)     # Explore by picking a random action
        else:
            action = np.argmax([Q[current_state][0],Q[current_state][1]]) # Exploit by picking the best action for this state
    
        next_state,reward = get_response(environment, current_state, action)
        
        q_curr = Q[current_state][action]   # get the value of the current q based on the next state and the action that we chose
        target_state = reward + gamma * np.max(Q[next_state])
        
        # update the Q table 
        Q[current_state][action] = (1-alpha) * q_curr + alpha * target_state
        
        # update the current state
        current_state = next_state    
        
        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay         

    return Q


#####################____TASK1____########################################
print("For environment 1 we get")   
print(implement_Q_learning(P1, 1000, 0.1, 0.9))
print("For environment 2 we get")   
print(implement_Q_learning(P2, 1000, 0.1, 0.9))


#####################____TASK2____########################################
#Generating environment P3\
P3 = generate_environment(10, 0.03)
print("For environment 3 we get")   
print(implement_Q_learning(P3, 1000, 0.1, 0.9))





















































#mr akis i love u