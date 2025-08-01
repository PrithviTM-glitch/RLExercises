import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class Graph:
    def __init__( self, env_name = "MountainCar-v0" ):
        self.env_name = env_name

    def plot_rewards(self, rewards, title = "Reward Over Episodes"):
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, label='Total Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_weights(self, weights, title = "Weights Over Time"):
        plt.figure(figsize=(10, 5))
        plt.plot(weights, label='Weights')
        plt.xlabel('Feature Index')
        plt.ylabel('Weight Value')
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_3d_surface(self, x, y, z, title = "3D Surface Plot"):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(title)
        plt.show()

    def plot_q_surface(self, q_value_func, weights, resolution=50):
        positions = np.linspace(-1.2, 0.6, resolution)
        velocities = np.linspace(-0.07, 0.07, resolution)
        X, Y = np.meshgrid(positions, velocities)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = (X[i, j], Y[i, j])
                Z[i, j] = max(q_value_func(state, a, weights) for a in [0, 1, 2])

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_zlabel("Max Q-Value")
        ax.set_title("Q-Value Surface")
        plt.tight_layout()
        plt.show()
    
    def plot_policy_heatmap(self, policy_func, weights, resolution=50):
        positions = np.linspace(-1.2, 0.6, resolution)
        velocities = np.linspace(-0.07, 0.07, resolution)
        X, Y = np.meshgrid(positions, velocities)
        policy_grid = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = (X[i, j], Y[i, j])
                policy_grid[i, j] = policy_func(state, weights, epsilon=0)

        plt.figure(figsize=(8, 6))
        sns.heatmap(policy_grid, xticklabels=False, yticklabels=False, cmap="viridis")
        plt.title("Preferred Action Across State Space")
        plt.xlabel("Position")
        plt.ylabel("Velocity")
        plt.tight_layout()
        plt.show()

    def plot_state_trajectory(self, positions, velocities, title="State Trajectory"):
        plt.figure(figsize=(8, 6))
        plt.plot(positions, velocities, marker='o', markersize=2)
        plt.xlabel("Position")
        plt.ylabel("Velocity")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self, rewards, window_size=100, title="Learning Curve"):
        """Plot rewards with moving average to show learning progress"""
        plt.figure(figsize=(12, 6))
        
        # Plot raw rewards
        plt.subplot(1, 2, 1)
        plt.plot(rewards, alpha=0.6, color='lightblue', label='Episode Rewards')
        
        # Calculate and plot moving average
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg, 
                    color='red', linewidth=2, label=f'Moving Average ({window_size} episodes)')
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Raw Rewards Over Time')
        plt.legend()
        plt.grid(True)
        
        # Plot cumulative average
        plt.subplot(1, 2, 2)
        cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        plt.plot(cumulative_avg, color='green', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Average Reward')
        plt.title('Cumulative Average Performance')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_weight_evolution(self, weight_history, feature_names=None, title="Weight Evolution"):
        """Plot how weights change over time during training"""
        weight_history = np.array(weight_history)
        
        if feature_names is None:
            feature_names = ['Position', 'Velocity', 'Action_0', 'Action_1', 'Action_2']
        
        plt.figure(figsize=(12, 8))
        
        for i in range(weight_history.shape[1]):
            plt.plot(weight_history[:, i], label=feature_names[i], linewidth=2)
        
        plt.xlabel('Training Step')
        plt.ylabel('Weight Value')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_td_error_history(self, td_errors, title="TD Error Over Time"):
        """Plot TD error to show learning convergence"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(td_errors, alpha=0.7, color='orange')
        plt.xlabel('Training Step')
        plt.ylabel('TD Error')
        plt.title('TD Error Over Time')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Plot moving average of absolute TD errors
        window_size = min(1000, len(td_errors) // 10)
        if window_size > 1:
            abs_errors = np.abs(td_errors)
            moving_avg = np.convolve(abs_errors, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(td_errors)), moving_avg, 
                    color='red', linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('|TD Error| (Moving Average)')
        plt.title(f'TD Error Magnitude (Window: {window_size})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_episode_lengths(self, episode_lengths, title="Episode Length Over Time"):
        """Plot how quickly the agent solves episodes (shorter = better)"""
        plt.figure(figsize=(10, 6))
        plt.plot(episode_lengths, alpha=0.7, color='purple')
        
        # Add moving average
        window_size = min(50, len(episode_lengths) // 10)
        if window_size > 1:
            moving_avg = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(episode_lengths)), moving_avg, 
                    color='red', linewidth=2, label=f'Moving Average ({window_size} episodes)')
        
        plt.xlabel('Episode')
        plt.ylabel('Steps to Complete')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_action_distribution(self, actions_taken, title="Action Distribution"):
        """Plot histogram of actions taken during training"""
        plt.figure(figsize=(8, 6))
        action_names = ['Push Left', 'No Push', 'Push Right']
        
        action_counts = np.bincount(actions_taken, minlength=3)
        bars = plt.bar(range(3), action_counts, color=['red', 'gray', 'blue'], alpha=0.7)
        
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.xticks(range(3), action_names)
        
        # Add percentage labels on bars
        total = sum(action_counts)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                    f'{action_counts[i]/total*100:.1f}%', 
                    ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_comprehensive_analysis(self, rewards, weight_history=None, td_errors=None, 
                                  episode_lengths=None, actions_taken=None):
        """Create a comprehensive dashboard of all training metrics"""
        fig = plt.figure(figsize=(16, 12))
        
        # Rewards
        plt.subplot(2, 3, 1)
        plt.plot(rewards, alpha=0.6, color='lightblue')
        window_size = min(100, len(rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', linewidth=2)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # Weight evolution
        if weight_history is not None:
            plt.subplot(2, 3, 2)
            weight_history = np.array(weight_history)
            feature_names = ['Pos', 'Vel', 'A0', 'A1', 'A2']
            for i in range(weight_history.shape[1]):
                plt.plot(weight_history[:, i], label=feature_names[i])
            plt.title('Weight Evolution')
            plt.xlabel('Training Step')
            plt.ylabel('Weight Value')
            plt.legend()
            plt.grid(True)
        
        # TD Error
        if td_errors is not None:
            plt.subplot(2, 3, 3)
            plt.plot(np.abs(td_errors), alpha=0.7, color='orange')
            plt.title('|TD Error| Over Time')
            plt.xlabel('Training Step')
            plt.ylabel('|TD Error|')
            plt.grid(True)
        
        # Episode lengths
        if episode_lengths is not None:
            plt.subplot(2, 3, 4)
            plt.plot(episode_lengths, alpha=0.7, color='purple')
            plt.title('Episode Lengths')
            plt.xlabel('Episode')
            plt.ylabel('Steps to Complete')
            plt.grid(True)
        
        # Action distribution
        if actions_taken is not None:
            plt.subplot(2, 3, 5)
            action_counts = np.bincount(actions_taken, minlength=3)
            plt.bar(range(3), action_counts, color=['red', 'gray', 'blue'], alpha=0.7)
            plt.title('Action Distribution')
            plt.xlabel('Action')
            plt.ylabel('Frequency')
            plt.xticks(range(3), ['Left', 'None', 'Right'])
        
        # Cumulative performance
        plt.subplot(2, 3, 6)
        cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        plt.plot(cumulative_avg, color='green', linewidth=2)
        plt.title('Cumulative Average Performance')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()