import numpy as np

def identify_straight_lines(path):

    if len(path) < 3:
        return True  

    vectors = np.diff(path, axis=0)
    
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    
    dot_products = np.abs(np.dot(normalized_vectors[:-1], normalized_vectors[1:].T))
    return np.all(dot_products > 0.99)  

def identify_circles_ellipses(path):
    if len(path) < 5:
        return None  

    center = np.mean(path, axis=0)
    
    distances = np.linalg.norm(path - center, axis=1)
    
    if np.allclose(distances, distances.mean(), rtol=0.05): 
        return 'circle'

    x, y = path[:, 0] - center[0], path[:, 1] - center[1]
    a, b = np.max(np.abs(x)), np.max(np.abs(y))
    ellipse_distances = np.sqrt((x/a)**2 + (y/b)**2)
    if np.allclose(ellipse_distances, 1, rtol=0.1):  
        return 'ellipse'
    
    return None

def identify_rectangles(path):
    if len(path) != 5 or not np.allclose(path[0], path[-1]):
        return False  

    vectors = np.diff(path[:-1], axis=0)
    
    dot_products = np.abs(np.dot(vectors, vectors.T))
    return (np.isclose(dot_products[0,2], 0) and 
            np.isclose(dot_products[1,3], 0) and 
            np.isclose(dot_products[0,1], 0) and 
            np.isclose(dot_products[2,3], 0))

def identify_regular_polygons(path):
   
    if not np.allclose(path[0], path[-1]):
        return 0 

    vectors = np.diff(path[:-1], axis=0)
    side_lengths = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle_diffs = np.diff(angles)
    
    if np.allclose(side_lengths, side_lengths.mean(), rtol=0.05) and \
       np.allclose(angle_diffs, angle_diffs.mean(), rtol=0.05):
        return len(path) - 1 
    
    return 0

def identify_star_shape(path):
   
    if len(path) < 6 or not np.allclose(path[0], path[-1]):
        return False  

    center = np.mean(path, axis=0)
    distances = np.linalg.norm(path - center, axis=1)
    
    long_short_pattern = np.abs(distances[:-1] - distances[1:]) > np.mean(distances) * 0.3
    return np.all(long_short_pattern[::2] != long_short_pattern[1::2])

def regularize_curves(paths):
    regularized = []
    for path in paths:
        if identify_straight_lines(path):
            regularized.append(('line', path))
        elif identify_circles_ellipses(path):
            regularized.append((identify_circles_ellipses(path), path))
        elif identify_rectangles(path):
            regularized.append(('rectangle', path))
        elif identify_regular_polygons(path):
            sides = identify_regular_polygons(path)
            regularized.append((f'regular_polygon_{sides}', path))
        elif identify_star_shape(path):
            regularized.append(('star', path))
        else:
            regularized.append(('irregular', path))
    return regularized