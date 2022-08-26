import numpy as np


def generate_fixed_position():
    """
    Generate fixed state and action position
    state_pos: numpy array, [81 x 2] shape [[10, 10], [10, 20], ..., [90, 90]]
    action_pos: numpy array, [81 x 2] shape [[10, 10], [10, 20], ..., [90, 90]]
    :return: state_pos, action_pos
    """
    state_pos = np.zeros((81, 2), dtype=np.int64)
    action_pos = np.zeros((81, 2), dtype=np.int64)
    for itr in range(state_pos.shape[0]):
        state_pos[itr] = [1 * (itr // 9 + 1), 1 * (itr % 9 + 1)]
        action_pos[itr] = [1 * (itr // 9 + 1), 1 * (itr % 9 + 1)]
    return state_pos, action_pos


def generate_random_position(domain_range, I, J, epsilon):
    """
    Generate random state and action position
    :param domain_range: domain range
    :param I: number of state
    :param J: number of action
    :return: state_pos, action_pos
    """
    state_pos = np.random.uniform(domain_range[0] + epsilon, domain_range[1] - epsilon, (I, 2))
    action_pos = np.random.uniform(domain_range[0] + epsilon, domain_range[1] - epsilon, (J, 2))
    return state_pos, action_pos


def generate_uniform_position(domain_range, num_state_x, num_state_y, num_actions, epsilon):
    x1 = np.linspace(domain_range[0] + epsilon, domain_range[1] - epsilon, num_state_x)
    x2 = np.linspace(domain_range[0] + epsilon, domain_range[1] - epsilon, num_state_y)
    xv, yv = np.meshgrid(x1, x2)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    state_pos = np.stack([xv, yv], axis=1)
    action_pos = np.random.uniform(domain_range[0] + epsilon, domain_range[1] - epsilon, (num_actions, 2))
    return state_pos, action_pos
