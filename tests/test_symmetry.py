import unittest
import numpy as np
from src.symmetry.symmetry import (
    find_reflection_symmetry,
    find_rotational_symmetry,
    analyze_symmetry
)

class TestSymmetry(unittest.TestCase):

    def test_find_reflection_symmetry(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        symmetries = find_reflection_symmetry(square)
        self.assertEqual(len(symmetries), 4)  # A square has 4 lines of symmetry

    def test_find_rotational_symmetry(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        order = find_rotational_symmetry(square)
        self.assertEqual(order, 4)  # A square has 4-fold rotational symmetry

    def test_analyze_symmetry(self):
        paths = [
            np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),  # square
            np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])  # equilateral triangle
        ]
        symmetries = analyze_symmetry(paths)
        self.assertEqual(len(symmetries), 2)
        self.assertEqual(symmetries[tuple(map(tuple, paths[0]))]['rotational_order'], 4)
        self.assertEqual(symmetries[tuple(map(tuple, paths[1]))]['rotational_order'], 3)

if __name__ == '__main__':
    unittest.main()