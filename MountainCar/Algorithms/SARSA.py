# Enhanced TD Learning with comprehensive data collection for visualization

import numpy as np
import gymnasium as gym
import time
from Helper.graph import Graph

env = gym.make("MountainCar-v0")  # Remove render_mode for training

def feature_vector(state, action): 
    pos, vel = state
    pos = (pos + 1.2) / 1.8   
    vel = (vel + 0.07) / 0.14
    feature_vector = [pos, vel, 
                     1 if action == 0 else 0, 
                     1 if action == 1 else 0, 
                     1 if action == 2 else 0]
    return np.array(feature_vector)

def q_value(state, action, weights): 
    phi = feature_vector(state, action)
    return np.dot(phi, weights)

def td_error(q_value, next_q_value, reward, gamma):
    return reward + gamma * next_q_value - q_value

def update_weights(weights, td_error, alpha, feature_vector):
    return weights + alpha * td_error * feature_vector

def policy(state, weights, epsilon=0.5):
    if np.random.rand() < epsilon: 
        return np.random.choice([0, 1, 2])
    else:
        q_values = [q_value(state, a, weights) for a in [0, 1, 2]]
        return np.argmax(q_values)

def main():
    num_episodes = int(input("Enter the number of episodes you want to run: "))
    alpha = int(input("Enter the learning rate (alpha * 1000): ")) / 1000.0
    decay = int(input("Enter the decay rate (0-100): ")) / 100.0
    gamma = 0.99
    weights = np.zeros(5)
    
    # Data collection for visualization
    total_rewards = []
    weight_history = []
    td_error_history = []
    episode_lengths = []
    actions_taken = []
    state_trajectories = []  # Store trajectories from sample episodes
    
    # Track epsilon decay
    initial_epsilon = 1
    epsilon_decay = 0.995
    min_epsilon = 0.01
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        
        # Store trajectory for sample episodes
        positions, velocities = [], []
        if episode % (num_episodes // 5) == 0:  # Store 5 sample trajectories
            store_trajectory = True
        else:
            store_trajectory = False
        
        # Decay learning rate and epsilon
        # current_alpha = alpha / (1 + decay * episode)

        # Decay exponentially
        current_alpha = max( alpha * (0.9995 ** (episode // 10)), 0.01)
        current_epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay ** (episode )))
        
        while not done: 
            if store_trajectory:
                positions.append(state[0])
                velocities.append(state[1])
            
            action = policy(state, weights, epsilon=current_epsilon)
            actions_taken.append(action)
            
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            next_action = policy(next_state, weights, epsilon=current_epsilon)  
            
            q_val = q_value(state, action, weights)
            next_q_value = q_value(next_state, next_action, weights)
            td_err = td_error(q_val, next_q_value, reward, gamma)
            feature_vec = feature_vector(state, action)
            
            # Store data for visualization
            td_error_history.append(td_err)
            weight_history.append(weights.copy())
            
            # Update weights
            weights = update_weights(weights, td_err, current_alpha, feature_vec)
            
            state = next_state
            total_reward += reward
            episode_length += 1
            
        total_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        
        if store_trajectory:
            state_trajectories.append((positions, velocities, f"Episode {episode}"))
        
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
            print(f"Episode {episode}: Avg Reward (last 100) = {avg_reward:.2f}, "
                  f"Alpha = {current_alpha:.4f}, Epsilon = {current_epsilon:.3f}")

    env.close()
    
    # Final statistics
    print(f"\nTraining Complete!")
    print(f"Final Average Reward (last 100): {np.mean(total_rewards[-100:]):.2f}")
    print(f"Final Weights: {weights}")
    
    # Create visualizations
    graph = Graph()
    print("\nGenerating visualizations...")
    
    # 1. Learning curve
    graph.plot_learning_curve(total_rewards, title="TD Learning Progress")
    
    # 2. Weight evolution
    graph.plot_weight_evolution(weight_history, title="Weight Evolution During Training")
    
    # 3. TD error history
    graph.plot_td_error_history(td_error_history, title="TD Error Convergence")
    
    # 4. Episode lengths
    graph.plot_episode_lengths(episode_lengths, title="Episode Length Over Time")
    
    # 5. Action distribution
    graph.plot_action_distribution(actions_taken, title="Action Distribution During Training")
    
    # 6. Q-value surface
    graph.plot_q_surface(q_value, weights, resolution=50)
    
    # 7. Policy heatmap
    graph.plot_policy_heatmap(policy, weights, resolution=50)
    
    # 8. State trajectories
    for positions, velocities, title in state_trajectories[:3]:  # Show first 3
        graph.plot_state_trajectory(positions, velocities, title=title)
    
    # 9. Comprehensive dashboard
    graph.plot_comprehensive_analysis(
        total_rewards, weight_history, td_error_history, 
        episode_lengths, actions_taken
    )
    
    return weights, total_rewards, weight_history

if __name__ == "__main__":
    main()
