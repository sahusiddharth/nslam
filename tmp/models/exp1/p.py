def parse_training_info(filename):
  """
  Parses a training log file and returns lists of extracted information.

  Args:
    filename: The name of the training log file.

  Returns:
    A tuple containing seven lists:
      - num_timesteps: List of number of timesteps for each entry.
      - mean_reward: List of mean reward for each entry.
      - median_reward: List of median reward for each entry.
      - mean_episodic_rewards: List containing lists of (mean) episodic rewards for each entry.
      - median_episodic_rewards: List containing lists of (median) episodic rewards for each entry.
      - min_episodic_rewards: List containing lists of (min) episodic rewards for each entry.
      - max_episodic_rewards: List containing lists of (max) episodic rewards for each entry.
  """
  num_timesteps = []
  mean_reward = []
  median_reward = []
  mean_episodic_rewards = []
  median_episodic_rewards = []
  min_episodic_rewards = []
  max_episodic_rewards = []


  with open(filename, 'r') as f:
    for line in f:
      if line.startswith("INFO:root:Time"):
        continue  # Skip time information
      elif "num timesteps" in line:
        # Extract number of timesteps
        timesteps = int(line.split()[3])
        num_timesteps.append(timesteps)
      elif "Rewards: Global step mean/med rew" in line:
        # Extract mean and median reward
        data = line.split(": ")[-2].split("/")
        data = line.split(": ")[-2].split(",")
        print(data)
        mean = float(data[0])
        median = float(data[1])
        mean_reward.append(mean)
        median_reward.append(median)
        if "Global eps mean/med/min/max eps rew" in line:
            # Extract episodic reward statistics
            data = line.split(": ")[-2].split("/")
            print(data)
            mean = float(data[0])
            median = float(data[1])
            min = float(data[2])
            max = float(data[3])
            mean_episodic_rewards.append(mean)
            median_episodic_rewards.append(median)
            min_episodic_rewards.append(min)
            max_episodic_rewards.append(max)

  return num_timesteps, mean_reward, median_reward, mean_episodic_rewards, median_episodic_rewards, min_episodic_rewards, max_episodic_rewards

# Example usage
filename = "train.txt"
num_timesteps, mean_reward, median_reward, mean_episodic_rewards, median_episodic_rewards, min_episodic_rewards, max_episodic_rewards = parse_training_info(filename)

# print("Number of Timesteps:", num_timesteps)
# print("Mean Reward:", mean_reward)
# print("Median Reward:", median_reward)
# print("Mean Episodic Rewards:", mean_episodic_rewards)
# print("Median Episodic Rewards:", median_episodic_rewards)
# print("Min Episodic Rewards:", min_episodic_rewards)
# print("Max Episodic Rewards:", max_episodic_rewards)
