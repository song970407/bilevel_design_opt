from src.solver.Implicit import ImplicitSolver
from src.solver.Single_layer import SingleLayerSolver
from src.solver.Genetic import GeneticSolver
from src.solver.CMA_ES import CMAESSolver


def get_implicit_solver(solver_config):
    return ImplicitSolver(solver_config)


def get_single_layer_solver(solver_config):
    return SingleLayerSolver(solver_config)


def get_genetic_solver(solver_config):
    return GeneticSolver(solver_config)

def get_cma_es_solver(solver_config):
    return CMAESSolver(solver_config)


solver_name_dict = {
    'implicit': get_implicit_solver,
    'single_layer': get_single_layer_solver,
    'genetic': get_genetic_solver,
    'cma_es': get_cma_es_solver
}


def get_solver(solver_name, solver_config):
    return solver_name_dict[solver_name](solver_config)
