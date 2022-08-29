from src.solver.Implicit import ImplicitSolver
from src.solver.Single_layer import SingleLayerSolver
from src.solver.Genetic import GeneticSolver


def get_implicit_solver(solver_config):
    return ImplicitSolver(solver_config)


def get_single_layer_solver(solver_config):
    return SingleLayerSolver(solver_config)


def get_genetic_solver(solver_config):
    return GeneticSolver(solver_config)


solver_name_dict = {
    'Implicit': get_implicit_solver,
    'Single_layer': get_single_layer_solver,
    'Genetic': get_genetic_solver
}


def get_solver(solver_name, solver_config):
    return solver_name_dict[solver_name](solver_config)
