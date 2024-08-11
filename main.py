from src.utils.io import read_csv
from src.utils.visualization import plot, polylines2svg
from src.regularization.regularize import regularize_curves
from src.symmetry.symmetry import analyze_symmetry
from src.completion.complete import handle_occlusions

def main():
    input_paths = read_csv('examples/isolated.csv')
    
    regularized_paths = regularize_curves(input_paths)
    
    symmetries = analyze_symmetry(regularized_paths)
    
    completed_paths = handle_occlusions(regularized_paths)
    
    plot(completed_paths)
    polylines2svg(completed_paths, 'output.svg')

if __name__ == "__main__":
    main()