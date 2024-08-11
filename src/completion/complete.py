import numpy as np
from scipy.interpolate import splprep, splev

def complete_curve(incomplete_path, occluding_path, num_points=100):

    intersections = find_intersections(incomplete_path, occluding_path)
    
    if len(intersections) != 2:
        return None  
    
    segment1, segment2 = split_path(incomplete_path, intersections)
    
    tck1, u1 = fit_spline(segment1)
    tck2, u2 = fit_spline(segment2)
    
    t = np.linspace(0, 1, num_points)
    completed_curve = np.vstack([
        splev(t[:num_points//2], tck1),
        splev(t[num_points//2:], tck2)
    ]).T
    
    return completed_curve

def find_intersections(path1, path2):
    
    intersections = []
    for i in range(len(path1) - 1):
        for j in range(len(path2) - 1):
            intersection = line_intersection(path1[i], path1[i+1], path2[j], path2[j+1])
            if intersection is not None:
                intersections.append(intersection)
    return intersections

def line_intersection(p1, p2, p3, p4):
  
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0:
        return None
    
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: 
        return None
    
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1:  
        x = x1 + ua * (x2-x1)
        y = y1 + ua * (y2-y1)
    return np.array([x, y])

def split_path(path, split_points):
    idx1 = np.argmin(np.linalg.norm(path - split_points[0], axis=1))
    idx2 = np.argmin(np.linalg.norm(path - split_points[1], axis=1))
    
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    
    segment1 = np.vstack((path[:idx1+1], split_points[0]))
    segment2 = np.vstack((split_points[1], path[idx2:]))
    
    return segment1, segment2

def fit_spline(points, k=3):
    tck, u = splprep(points.T, k=k, s=0)
    return tck, u

def handle_occlusions(paths):
    completed_paths = []
    for i, path in enumerate(paths):
        completed = False
        for j, other_path in enumerate(paths):
            if i != j:
                completed_path = complete_curve(path, other_path)
                if completed_path is not None:
                    completed_paths.append(completed_path)
                    completed = True
                    break
        if not completed:
            completed_paths.append(path)
    return completed_paths