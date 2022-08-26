from typing import List


def generate_target_trajectory(target_values: List[float], target_times: List[int]):
    assert len(target_values) == len(target_times) + 1

    entire_time = sum(target_times)

    ref_temps = []

    for step_idx in range(len(target_times)):
        for time_idx in range(target_times[step_idx]):
            ref_temps.append(
                target_values[step_idx] + (time_idx + 1) * (target_values[step_idx + 1] - target_values[step_idx]) /
                target_times[step_idx])
    return ref_temps
