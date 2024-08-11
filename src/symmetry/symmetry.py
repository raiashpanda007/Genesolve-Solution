import numpy as np
from scipy.spatial.distance import cdist

def find_reflection_symmetry(path, tolerance=0.05):
    
    symmetry_axes = []
    n_points = len(path)
    
    for i in range(n_points):
        for j in range(i+1, n_points):
            axis_point = (path[i] + path[j]) / 2
            axis_vector = path[j] - path[i]
            normal_vector = np.array([-axis_vector[1], axis_vector[0]])
            normal_vector /= np.linalg.norm(normal_vector)
            
            reflected_points = reflect_points(path, axis_point, normal_vector)
            
            distances = cdist(path, reflected_points)
            min_distances = np.min(distances, axis=1)
            
            if np.all(min_distances < tolerance * np.linalg.norm(axis_vector)):
                symmetry_axes.append((axis_point, normal_vector))
    
    return symmetry_axes

def reflect_points(points, axis_point, normal_vector):
    return points - 2 * np.outer(
        np.dot(points - axis_point, normal_vector),
        normal_vector
    )

def find_rotational_symmetry(path, tolerance=0.05):
    center = np.mean(path, axis=0)
    centered_path = path - center
    
    max_order = len(path) // 2
    for order in range(2, max_order + 1):
        angle = 2 * np.pi / order
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        rotated_path = np.dot(centered_path, rotation_matrix.T) + center
        
        if np.all(cdist(path, rotated_path).min(axis=1) < tolerance * np.ptp(path)):
            return order
    
    return 1

def analyze_symmetry(paths):
   
    symmetries = {}
    for path in paths:
        reflection_axes = find_reflection_symmetry(path)
        rotational_order = find_rotational_symmetry(path)
        
        symmetries[tuple(map(tuple, path))] = {
            'reflection_axes': reflection_axes,
            'rotational_order': rotational_order
        }
    
    return symmetries