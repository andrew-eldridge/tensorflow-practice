# Machine Learning Model
# Written by Andrew Eldridge
# 4/9/2020

import gym
import numpy as np
import matplotlib.pyplot as plt


MAX_STEPS = 100  # max number of steps allowed in each environment
LEARNING_RATE = 0.81  # dictates how quickly Q-table is populated
GAMMA = 0.96  # weight given to future actions/states
RENDER = False  # toggles environment rendering


# get average of array
def get_average(values):
    return sum(values)/len(values)


# plot training results
def plot_training_results(rewards):
    avg_rewards = []
    for i in range(0, len(rewards), 1000):
        avg_rewards.append(get_average(rewards[i:i + 1000]))  # get reward averages at 1000-reward increments
    plt.plot(avg_rewards)
    plt.yticks(np.arange(0.1, max(avg_rewards)+0.1, 0.1))
    plt.ylabel('Average reward')
    plt.xlabel('Trials (thousands)')
    plt.show()


# print training results
def print_training_results(num_trials, q, rewards):
    print("\n----------------------------------------------------------------------")
    print(f"Q-table after {num_trials} training episodes:\n{q}")  # print the Q-table
    print(f"Average reward: {sum(rewards) / len(rewards)}")  # print average reward
    print("----------------------------------------------------------------------\n")
    plot_training_results(rewards)


# print test results
def print_test_results(training_period, num_success):
    # print results of initial test
    print("\n----------------------------------------------------------------------")
    print(f"Number of successful navigations {training_period}: {num_success}/1000")
    print(f"Success ratio: {num_success / 1000}")
    print("----------------------------------------------------------------------\n")


# train ML model
def train_model(env, episodes, q, epsilon):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        for _ in range(MAX_STEPS):
            if RENDER:
                env.render()  # optional rendering

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # decides a random action based on current state
            else:
                action = np.argmax(q[state, :])  # determines the best action from Q-table given current state

            next_state, reward, done, _ = env.step(action)

            q[state, action] = q[state, action] + LEARNING_RATE * (
                        reward + GAMMA * np.max(q[next_state, :]) - q[state, action])

            state = next_state

            if done:
                rewards.append(reward)
                epsilon -= 0.001
                break
    return rewards, q, epsilon


# test ML model
def test_model(env, episodes, q):
    num_success = 0
    for episode in range(episodes):
        state = env.reset()
        for _ in range(MAX_STEPS):
            action = np.argmax(q[state, :])  # choose action based on current model
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                if reward == 1:
                    num_success += 1
                break
    return num_success


# main method
def main():
    env = gym.make('FrozenLake-v0')  # initialize environment
    states = env.observation_space.n  # number of possible states
    actions = env.action_space.n  # number of actions in a state
    q = np.zeros((states, actions))  # initialize Q-table
    epsilon = 0.9  # proportion of random vs informed moves in training

    # run test on model before teaching it anything
    num_success = test_model(env, 1000, q)  # test the model pre-training over 1000 episodes
    print_test_results("pre-training", num_success)  # print results of initial test

    # train the model
    rewards, q, epsilon = train_model(env, 10000, q, epsilon)  # train the model over 10000 episodes
    print_training_results(10000, q, rewards)  # print/plot results of first training

    # run another test after training
    num_success = test_model(env, 1000, q)  # test the model post-training 1 over 1000 episodes
    print_test_results("post-training 1", num_success)  # print results of post-training 1 test

    # train one more time
    rewards, q, epsilon = train_model(env, 10000, q, epsilon)  # train the model over 10000 more episodes (20000 total)
    print_training_results(20000, q, rewards)  # print/plot results of second training

    # run a final test
    num_success = test_model(env, 1000, q)  # test the model post-training 2 over 1000 episodes
    print_test_results("post-training 2", num_success)  # print results of post-training 2 test


main()
