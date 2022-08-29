from src.solver.Implicit import *
from src.solver.Single_layer import *
from src.solver.Genetic import *


def get_implicit_solver(solver_config):
    return


def get_single_layer_solver(solver_config):
    return


def get_genetic_solver(solver_config):
    return


solver_name_dict = {
    'Implicit': get_implicit_solver,
    'Single_layer': get_single_layer_solver,
    'Genetic': get_genetic_solver
}


def get_solver(solver_name, solver_config):
    return solver_name_dict[solver_name](solver_config)
