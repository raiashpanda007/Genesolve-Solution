import unittest
import numpy as np

from src.regularization.regularize import (
    identify_straight_lines,
    identify_circles_ellipses,
    identify_rectangles,
    identify_regular_polygons,
    identify_star_shape,
    regularize_curves
)

class TestRegularization(unittest.TestCase):

    def test_identify_straight_lines(self):
        line = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        self.assertTrue(identify_straight_lines(line))
        
        not_line = np.array([[0, 0], [1, 1], [2, 2], [3, 4]])
        self.assertFalse(identify_straight_lines(not_line))
        
        vertical_line = np.array([[1, 0], [1, 1], [1, 2], [1, 3]])
        self.assertTrue(identify_straight_lines(vertical_line))
        
        horizontal_line = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
        self.assertTrue(identify_straight_lines(horizontal_line))

    def test_identify_circles_ellipses(self):
        circle = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
        self.assertEqual(identify_circles_ellipses(circle), 'circle')
        
        ellipse = np.array([[2, 0], [0, 1], [-2, 0], [0, -1], [2, 0]])
        self.assertEqual(identify_circles_ellipses(ellipse), 'ellipse')
        
        not_circle_ellipse = np.array([[0, 0], [1, 1], [2, 0], [1, -1], [0, 0]])
        self.assertIsNone(identify_circles_ellipses(not_circle_ellipse))
        
        small_circle = np.array([[0.1, 0], [0, 0.1], [-0.1, 0], [0, -0.1], [0.1, 0]])
        self.assertEqual(identify_circles_ellipses(small_circle), 'circle')

    def test_identify_rectangles(self):
        rectangle = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]])
        self.assertTrue(identify_rectangles(rectangle))
        
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        self.assertTrue(identify_rectangles(square))
        
        not_rectangle = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [0, 0]])
        self.assertFalse(identify_rectangles(not_rectangle))
        
        rotated_rectangle = np.array([[0, 0], [1, 1], [0, 2], [-1, 1], [0, 0]])
        self.assertTrue(identify_rectangles(rotated_rectangle))

    def test_identify_regular_polygons(self):
        triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
        self.assertEqual(identify_regular_polygons(triangle), 3)
        
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        self.assertEqual(identify_regular_polygons(square), 4)
        
        pentagon = np.array([
            [1, 0], [0.309, 0.951], [-0.809, 0.588], 
            [-0.809, -0.588], [0.309, -0.951], [1, 0]
        ])
        self.assertEqual(identify_regular_polygons(pentagon), 5)
        
        not_regular = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]])
        self.assertEqual(identify_regular_polygons(not_regular), 0)

    def test_identify_star_shape(self):
        star = np.array([
            [0, 1], [0.2, 0.2], [1, 0], [0.2, -0.2], 
            [0, -1], [-0.2, -0.2], [-1, 0], [-0.2, 0.2], [0, 1]
        ])
        self.assertTrue(identify_star_shape(star))
        
        not_star = np.array([[0, 0], [1, 1], [2, 0], [1, -1], [0, 0]])
        self.assertFalse(identify_star_shape(not_star))
        
        complex_star = np.array([
            [0, 1], [0.2, 0.2], [1, 0.3], [0.3, 0.1], [0.5, -0.5],
            [0, -0.2], [-0.5, -0.5], [-0.3, 0.1], [-1, 0.3], [-0.2, 0.2], [0, 1]
        ])
        self.assertTrue(identify_star_shape(complex_star))

    def test_regularize_curves(self):
        paths = [
            np.array([[0, 0], [1, 1], [2, 2]]),  # line
            np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]]),  # circle
            np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]]),  # rectangle
            np.array([[0, 1], [0.2, 0.2], [1, 0], [0.2, -0.2], 
                      [0, -1], [-0.2, -0.2], [-1, 0], [-0.2, 0.2], [0, 1]])  # star
        ]
        regularized = regularize_curves(paths)
        self.assertEqual(len(regularized), 4)
        self.assertEqual(regularized[0][0], 'line')
        self.assertEqual(regularized[1][0], 'circle')
        self.assertEqual(regularized[2][0], 'rectangle')
        self.assertEqual(regularized[3][0], 'star')

    def test_edge_cases(self):
        tiny_circle = np.array([[0.01, 0], [0, 0.01], [-0.01, 0], [0, -0.01], [0.01, 0]])
        self.assertEqual(identify_circles_ellipses(tiny_circle), 'circle')

        almost_square = np.array([[0, 0], [1, 0], [1.01, 1], [0, 1], [0, 0]])
        self.assertTrue(identify_rectangles(almost_square))
        self.assertEqual(identify_regular_polygons(almost_square), 0) 
        

        large_line = np.array([[0, 0], [1e6, 1e6], [2e6, 2e6]])
        self.assertTrue(identify_straight_lines(large_line))

    def test_noise_resistance(self):
        t = np.linspace(0, 2*np.pi, 100)
        circle = np.column_stack((np.cos(t), np.sin(t)))
        noisy_circle = circle + np.random.normal(0, 0.05, circle.shape)
        self.assertEqual(identify_circles_ellipses(noisy_circle), 'circle')

        star = np.array([
            [0, 1], [0.2, 0.2], [1, 0], [0.2, -0.2], 
            [0, -1], [-0.2, -0.2], [-1, 0], [-0.2, 0.2], [0, 1]
        ])
        noisy_star = star + np.random.normal(0, 0.05, star.shape)
        self.assertTrue(identify_star_shape(noisy_star))

if __name__ == '__main__':
    unittest.main()