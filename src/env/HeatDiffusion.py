from typing import List

import numpy as np
from fenics import *
from mshr import *
from scipy import signal
from src.utils.ou_process import ou_process_clip
np.random.seed(0)


class HeatDiffusionSystem:
    def __init__(self,
                 num_cells: int,
                 dt: float,
                 epsilon: float,
                 domain_range: [-1.0, 1.0],
                 action_min: float,
                 action_max: float,
                 state_pos: np.array,  # [I x 2] numpy integer array
                 action_pos: np.array):  # [J x 2] numpy integer array
        # Furnace Setting
        self.num_cells = num_cells
        self.dt = dt
        self.epsilon = epsilon
        self.domain_range = domain_range
        self.action_min = action_min
        self.action_max = action_max
        self.state_dim = state_pos.shape[0]
        self.action_dim = action_pos.shape[0]
        self.state_pos = state_pos
        self.action_pos = action_pos
        self.reset()

    def reset(self):
        self.domain = Rectangle(Point(self.domain_range[0], self.domain_range[0]),
                                Point(self.domain_range[1], self.domain_range[1]))
        self.mesh = generate_mesh(self.domain, self.num_cells)
        self.V = FunctionSpace(self.mesh, 'P', 1)

        self.bc = DirichletBC(self.V, Constant(0), DomainBoundary())

        # Define initial value
        self.u_0 = Constant(0)
        self.u_n = interpolate(self.u_0, self.V)

        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        self.u = Function(self.V)
        self.t = 0.0
        self.state_trajectory = []
        self.action_trajectory = []
        obs = self.observe()
        self.state_trajectory.append(obs)

    def observe(self):
        obs = []
        for pos in self.state_pos:
            obs.append(self.u(pos))
        obs = np.array(obs)
        return obs

    def generate_f(self, a):
        f_list = ['0.0']
        for k, pos in enumerate(self.action_pos):
            f_list.append('+{}*({}<=x[0] & x[0]<= {} & {}<=x[1] & x[1]<={})'.format(a[k],
                                                                                    pos[0] - self.epsilon,
                                                                                    pos[0] + self.epsilon,
                                                                                    pos[1] - self.epsilon,
                                                                                    pos[1] + self.epsilon))
        return ''.join(map(str, f_list))

    def step(self, action):
        self.f = Expression(self.generate_f(action), degree=2)
        self.u = TrialFunction(self.V)
        self.F = (self.u) * (self.v) * dx + (self.dt) * dot(grad(self.u), grad(self.v)) * dx - (
                    (self.u_n) + (self.dt) * (self.f)) * (self.v) * dx
        self.a, self.L = lhs(self.F), rhs(self.F)
        self.A, self.b = assemble_system(self.a, self.L, self.bc)
        self.t += self.dt
        self.u = Function(self.V)
        solve(self.A, self.u.vector(), self.b)
        self.u_n.assign(self.u)
        obs = self.observe()
        self.state_trajectory.append(obs)
        self.action_trajectory.append(action)
        return obs

    def return_trajectory(self):
        return np.stack(self.state_trajectory, axis=0), np.stack(self.action_trajectory, axis=0)

    def generate_random_trajectory(self, T):
        """
        :param T: int
        :return:
        """
        self.reset()
        action = np.zeros((T, self.action_dim))
        for k in range(self.action_dim):
            action[:, k] = ou_process_clip(len=T)
        action = action * (self.action_max - self.action_min) / 2 + (self.action_max - self.action_min) / 2
        for t in range(T):
            # print('Timestep [{}] / [{}]'.format(t, T))
            self.step(action[t])
        state_trajectory, action_trajectory = self.return_trajectory()
        return state_trajectory, action_trajectory
