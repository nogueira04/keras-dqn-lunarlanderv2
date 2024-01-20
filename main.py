from agent import Agent
from utils import plot_learning
import numpy as np
import tensorflow as tf
import gym

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    env = gym.make("LunarLander-v2")
    lr = 0.001
    n_games = 500
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, input_dims=env.observation_space.shape,
                   n_actions=env.action_space.n, mem_size=1000000, batch_size=64, epsilon_end=0.01)
    
    scores, eps_history = [], []

    for i in range(n_games):
        done = False
        score = 0
        observation, _ = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
            
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(f"Episode {i}: score = {score}, avg_score = {avg_score}, epsilon = {agent.epsilon}")

    filename = "lunarlander_tf2.png"
    x = [i + 1 in range(n_games)]
    plot_learning(x, scores, eps_history, filename)