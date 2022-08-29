def minmax_scale(traj, original_scaler, scaled_scaler):
    ratio = (traj - original_scaler[0]) / (original_scaler[1] - original_scaler[0])
    return scaled_scaler[0] + (scaled_scaler[1] - scaled_scaler[0]) * ratio
