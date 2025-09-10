"""
 This is an implementation of the MonteCarlo method on action value function approximation 
 and learning an epsilon greedy policy in terms of this.

"""


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np 
import gymnasium as gym
import time 
from Helper.graph import Graph

# Initialise the environment
# env = gym.make("MountainCar-v0", render_mode = "human")
env = gym.make("MountainCar-v0")

def feature_vector(state, action):
    pos, vel = state

    # Normalised position and velocity values 
    # pos = max(0, min(1, (pos + 1.2) / 1.8))
    # vel = max(0, min(1, (vel + 0.07) / 0.14))

    feature_vector = [
        pos, vel, pos**2, vel**2, pos*vel, 
        1 if action == 0 else 0,
        1 if action == 1 else 0,
        1 if action == 2 else 0
    ]
    return np.array(feature_vector)

def q_value(state, action, weights):
    phi = feature_vector(state, action)
    return np.dot(phi, weights)

def mc_error(actual_return, state, action, weights):
    return actual_return - q_value(state, action, weights)

def update_weights(mc_error, weights, alpha, feature_vector):
    return weights + (alpha*mc_error* feature_vector)

def policy(state, weights, epsilon = 0.5):
    if np.random.rand() < epsilon:
        return np.random.choice([0,1,2])
    
    else: 
        q_values = [q_value(state, a, weights) for a in [0,1,2]]
        return np.argmax(q_values)
    
def main():
    num_episodes = int(input("Enter the number of episodes that you want to run: "))
    alpha = 0.25
    decay = 0.5

    gamma = 0.99

    weights = np.zeros(8)

    total_rewards = []
    weight_history = []
    mc_error_history = []
    episode_lengths = []
    actions_taken = []
    state_trajectories = [] 

    initial_epsilon = 1
    epsilon_decay = 0.995
    min_epsilon = 0.4

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        positions, velocities = [], []
        store_trajectory = (episode % max(1, num_episodes // 5) == 0)
        current_alpha = max(alpha * (0.9995 ** (episode // 10)), 0.01)
        current_epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay ** (episode)))
        episode_data = []
        # Collect episode
        while not done:
            if store_trajectory:
                positions.append(state[0])
                velocities.append(state[1])
            action = policy(state, weights, epsilon=current_epsilon)
            actions_taken.append(action)
            next_state, reward, done, truncated, info = env.step(action)
            position_bonus = 0.1 * max(0, next_state[0] + 1.2) 
            reward += position_bonus
            current_distance = abs(next_state[0] - 0.5)
            max_distance = abs(-1.2 - 0.5)  # Distance from leftmost position to goal

            # Reward for being closer to goal (normalized between 0 and some bonus value)
            distance_bonus = 0.1 * (1 - current_distance / max_distance)
            reward += distance_bonus
            done = done or truncated
            episode_data.append({
                'state': state.copy(),
                'action': action,
                'reward': reward
            })
            total_reward += reward
            episode_length += 1
            state = next_state
            
        if episode < 10 or episode % 100 == 0:
            print(f"Episode {episode}:")
            print(f"  Steps: {episode_length}, Total Reward: {total_reward}")
            print(f"  Final position: {state[0]:.4f} (goal at 0.5)")
            print(f"  Max position this episode: {max([data['state'][0] for data in episode_data]):.4f}")
            print(f"  Current weights: {weights}")
            
            # Test your policy - what does it prefer in different states?
            test_state = np.array([-0.5, 0.0])  # Middle position, no velocity
            q_vals = [q_value(test_state, a, weights) for a in [0,1,2]]
            print(f"  Q-values at test state {test_state}: {q_vals}")
        if store_trajectory:
            state_trajectories.append((positions, velocities, f"Episode {episode}"))
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards) if total_rewards else 0
            print(f"Episode {episode}: Avg Reward (last 100) = {avg_reward:.2f}, "
                  f"Alpha = {current_alpha:.4f}, Epsilon = {current_epsilon:.3f}")
        # MC update: use weights from start of episode for all steps
        returns = []
        G = 0
        for i in reversed(range(len(episode_data))):
            G = episode_data[i]['reward'] + gamma * G
            returns.append(G)
        returns.reverse()
        episode_errors = []
        # Use weights from start of episode for all updates
        weights_episode = weights.copy()
        for i in range(len(episode_data)):
            state_i = episode_data[i]['state']
            action_i = episode_data[i]['action']
            return_i = returns[i]
            error = mc_error(return_i, state_i, action_i, weights_episode)
            episode_errors.append(error)
            phi = feature_vector(state_i, action_i)
            weights_episode = update_weights(error, weights_episode, current_alpha, phi)
        weights = weights_episode
        # Store metrics for analysis (per episode)
        total_rewards.append(total_reward)
        weight_history.append(weights.copy())
        mc_error_history.append(np.mean(np.abs(episode_errors)))
        episode_lengths.append(episode_length)

    print("\nTraining Complete!")
    print(f"Final Average Reward (last 100): {np.mean(total_rewards[-100:]):.2f}")
    print(f"Final Weights: {weights}")

    # --- Graphing Section ---
    graph = Graph(env_name="MountainCar-v0")
    # Rewards and learning curve
    graph.plot_learning_curve(total_rewards, window_size=100, title="MC: Learning Curve")
    # Weights evolution
    graph.plot_weight_evolution(weight_history, title="MC: Weight Evolution")
    # MC error history
    graph.plot_td_error_history(mc_error_history, title="MC: MC Error Over Time")
    # Episode lengths
    graph.plot_episode_lengths(episode_lengths, title="MC: Episode Lengths")
    # Action distribution
    graph.plot_action_distribution(actions_taken, title="MC: Action Distribution")
    # Q-value surface
    graph.plot_q_surface(q_value, weights, resolution=40)
    # Policy heatmap
    graph.plot_policy_heatmap(policy, weights, resolution=40)
    # State trajectories (sampled)
    for positions, velocities, label in state_trajectories:
        graph.plot_state_trajectory(positions, velocities, title=f"MC: {label}")
    # Comprehensive dashboard
    graph.plot_comprehensive_analysis(total_rewards, weight_history, mc_error_history, episode_lengths, actions_taken)

if __name__ == "__main__":
    main()

