import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances  # If distances need recomputation

def simulate_removal(df: pd.DataFrame, distance_matrix: np.ndarray, products_to_remove: list, threshold: float = None):
    """
    Simulate removal of products and find substitutes.
    
    Args:
        df: DataFrame with products (rows) and features (columns). Index should be product identifiers.
        distance_matrix: Precomputed NxN distance matrix (symmetric, where distance_matrix[i,j] is dist between product i and j).
        products_to_remove: List of product identifiers (matching df.index) to remove.
        threshold: Optional float; if provided, flag if mean substitute dist > threshold as a 'gap'.
    
    Returns:
        dict with:
            - 'substitute_distances': List of min distances for each removed product.
            - 'mean_dist': Mean of substitute distances.
            - 'max_dist': Max of substitute distances.
            - 'substitutes': Dict mapping removed product to its substitute (nearest remaining).
            - 'gaps': List of removed products with substitute dist > threshold (if threshold provided).
    """
    if not isinstance(distance_matrix, np.ndarray):
        raise ValueError("distance_matrix must be a NumPy array.")
    
    all_products = list(df.index)
    remaining_products = [p for p in all_products if p not in products_to_remove]
    
    if not remaining_products:
        return {'error': 'No remaining products after removal.'}
    
    substitute_distances = []
    substitutes = {}
    gaps = []
    
    for removed in products_to_remove:
        if removed not in all_products:
            continue  # Skip invalid
        removed_idx = all_products.index(removed)
        
        # Distances from removed to remaining
        remaining_idxs = [all_products.index(p) for p in remaining_products]
        dists_to_remaining = distance_matrix[removed_idx, remaining_idxs]
        
        min_dist_idx = np.argmin(dists_to_remaining)
        min_dist = dists_to_remaining[min_dist_idx]
        substitute = remaining_products[min_dist_idx]
        
        substitute_distances.append(min_dist)
        substitutes[removed] = {'substitute': substitute, 'distance': min_dist}
        
        if threshold and min_dist > threshold:
            gaps.append(removed)
    
    mean_dist = np.mean(substitute_distances) if substitute_distances else 0
    max_dist = np.max(substitute_distances) if substitute_distances else 0
    
    result = {
        'substitute_distances': substitute_distances,
        'mean_dist': mean_dist,
        'max_dist': max_dist,
        'substitutes': substitutes
    }
    if threshold:
        result['gaps'] = gaps
    
    return result
