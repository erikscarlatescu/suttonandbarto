import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    policy = np.load("policy.npy")
    policy = np.flip(policy, 0)
    plt.imshow(policy.T, cmap='summer', interpolation='nearest')
    plt.show()