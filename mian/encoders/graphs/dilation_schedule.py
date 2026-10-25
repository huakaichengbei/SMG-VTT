def tactile_k_schedule(n_layers=12,k_start=9,k_end=18):
    """Linearly increase k across all layers, inclusive endpoints (implementation choice)."""
    return [round(k_start+(k_end-k_start)*i/max(1,n_layers-1)) for i in range(n_layers)]

def tactile_dilation_schedule(n_layers=12):
    """For 12 layers, use four layers each with dilation 1, 2, 3."""
    return [min(3,i//max(1,n_layers//3)+1) for i in range(n_layers)]
