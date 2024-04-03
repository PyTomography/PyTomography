def compute_TEW(projection_lower, projection_upper, width_lower, width_upper, width_peak, return_scatter_variance_estimate=False):
    scatter_estimate = (projection_lower/width_lower + projection_upper/width_upper)*width_peak / 2
    if return_scatter_variance_estimate:
        scatter_variance_estimate = (width_peak / width_lower / 2) ** 2 * projection_lower + (width_peak / width_upper / 2) ** 2 * projection_upper
        return scatter_estimate, scatter_variance_estimate
    else:
        return scatter_estimate