import unittest
import numpy as np
from src.completion.complete import (
    complete_curve,
    find_intersections,
    split_path,
    handle_occlusions
)

class TestCompletion(unittest.TestCase):

    def test_find_intersections(self):
        path1 = np.array([[0, 0], [2, 2]])
        path2 = np.array([[0, 2], [2, 0]])
        intersections = find_intersections(path1, path2)
        self.assertEqual(len(intersections), 1)
        np.testing.assert_almost_equal(intersections[0], [1, 1])

    def test_split_path(self):
        path = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        split_points = [[1, 1], [2, 2]]
        segment1, segment2 = split_path(path, split_points)
        np.testing.assert_array_equal(segment1, [[0, 0], [1, 1]])
        np.testing.assert_array_equal(segment2, [[2, 2], [3, 3]])

    def test_complete_curve(self):
        incomplete_path = np.array([[0, 0], [1, 1], [2, 2]])
        occluding_path = np.array([[1, 0], [1, 2]])
        completed = complete_curve(incomplete_path, occluding_path)
        self.assertIsNotNone(completed)
        self.assertEqual(len(completed), 100)  

    def test_handle_occlusions(self):
        paths = [
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[1, 0], [1, 2]])
        ]
        completed_paths = handle_occlusions(paths)
        self.assertEqual(len(completed_paths), 2)
        self.assertGreater(len(completed_paths[0]), len(paths[0]))

if __name__ == '__main__':
    unittest.main()