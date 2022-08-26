import numpy as np
import matplotlib.pyplot as plt


def ou_process(len, theta=0.001, mu=0.0, sigma=10.0, dt=0.1):
    """
    Process Ornstein-Uhlenbeck process
    X_{t+1} = X_t - theta * (mu - X_t) * dt + sigma * sigma * dW_t
    :param len: length of ou process, int
    :param theta: float
    :param mu: float
    :param sigma: float
    :return: ou process X_t, [len] numpy.array
    """
    eta = np.random.normal(size=len)
    w = np.zeros(len + 1)
    for i in range(len):
        w[i + 1] = np.sum(eta[:i + 1]) / np.sqrt(i + 1)
    x = np.zeros(len + 1)
    x[0] = mu
    for i in range(len):
        x[i + 1] = x[i] + theta * dt * (mu - x[i]) + sigma * (w[i + 1] - w[i]) * dt
    x = x[1:]
    return x


def ou_process_clip(len, theta=0.01, mu=0.0, sigma=10.0, dt=0.1, clip_min=-1.0, clip_max=1.0):
    eta = np.random.normal(size=len)
    w = np.zeros(len + 1)
    for i in range(len):
        w[i + 1] = np.sum(eta[:i + 1]) / np.sqrt(i + 1)
    x = np.zeros(len + 1)
    x[0] = mu
    for i in range(len):
        x[i + 1] = np.clip(x[i] + theta * dt * (mu - x[i]) + sigma * (w[i + 1] - w[i]) * dt, clip_min, clip_max)
    x = x[1:]
    return x


if __name__ == '__main__':
    x = ou_process(len=100)
    plt.plot(x)
    y = ou_process_clip(len=100)
    plt.plot(y)
    plt.show()
