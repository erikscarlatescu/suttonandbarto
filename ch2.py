import numpy as np
import time

def value_function_sample_avg(samples):
    if len(samples) == 0: return 0
    return (np.sum(samples)) / samples.shape[0]

def value_function_bayesian(samples):
    if len(samples) == 0: return 0
    return (np.sum(samples)) / (samples.shape[0] + 1.0)

def append_value_estimate_sample_avg(values, counts, index, reward):
    values[index] *= (counts[index] / (counts[index] + 1.0))
    values[index] += (reward / (counts[index] + 1.0))
    counts[index] += 1

def epsilon_greedy(values, epsilon=0.1):
    n = values.shape[0]

    probs = np.ones(values.shape) * epsilon / (n - 1)

    # evenly distribute probabilities between possible maxes if we are just starting out
    if np.max(values) == 0:
        num_zero = len(np.where(values == 0)[0])
        if num_zero < n:
            probs[values == 0] = (1 - epsilon) / num_zero
            probs[values != 0] = epsilon / (n - num_zero)
        else:
            probs = np.ones(values.shape) / n
    else: # otherwise leave everything else alone and set one arm to have probability 1 - epsilon
        biggest = np.argmax(values)
        probs[biggest] = 1 - epsilon

    return probs

temp = 0.01
def softmax(values):
    e_v = np.exp( (values - values.max()) / temp)
    return e_v / e_v.sum()

# separate from bandit to emphasize that bandit has no information about the true means
class NArmMachine:
    def __init__(self, num_levers=10):
        self.num_levers = num_levers

        self.true_means = [0] * self.num_levers
        for i in range(self.num_levers):
            self.true_means[i] = np.random.normal()

    def pull_lever(self, i):
        return np.random.normal(loc=self.true_means[i])

class NArmBandit:
    def __init__(self, num_arms=10, value_function=append_value_estimate_sample_avg, prob_function=epsilon_greedy, MachineType=NArmMachine):
        self.num_arms = num_arms
        self.value_function = value_function
        self.prob_function = prob_function

        self.MachineType = MachineType
        self.machine = self.MachineType()
        self.values = np.zeros(self.num_arms)
        self.counts = np.zeros(self.num_arms)

        self.epsilon = 0.1

    def do_action_and_update(self):
        probs = self.prob_function(self.values, self.epsilon)
        choice = np.random.choice(self.num_arms, 1, p=probs)[0]

        reward = self.machine.pull_lever(choice)
        append_value_estimate_sample_avg(self.values, self.counts, choice, reward)

        return reward

    def perform_episode(self, episode_length=1000):
        self.machine = self.MachineType()
        self.values = np.zeros(self.num_arms)
        self.counts = np.zeros(self.num_arms)

        rewards = []
        for i in range(episode_length):
            # self.epsilon = max(0, 0.9 - i / 200.0)
            self.epsilon = 0.01
            rewards.append(self.do_action_and_update())

        return rewards

    def test_method(self, num_tests=400):
        rewards = []
        for i in range(num_tests):
            #start = time.time()
            rewards.append(self.perform_episode())
            #end = time.time()
            #print(f"Time to run is {end - start} seconds")

        return rewards


if __name__ == "__main__":
    bandit = NArmBandit()
    sampled_rewards = bandit.perform_episode()

    rewards = np.array(bandit.test_method())
    rewards = np.mean(rewards, axis=0)
    import matplotlib.pyplot as plt

    plt.plot(rewards)
    plt.show()

    print("Hey")
