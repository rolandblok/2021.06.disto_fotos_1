
from base_types import Coordinate_2D

class TwoDToTwoDPerspectiveTransformation:
    
    def __init__(self):
        self.__calibrated = False

    def calibrate(self, input_coordinates, output_coordinates):
        if len(input_coordinates) != len(output_coordinates):
            raise Exception("Expect to get the same number of input and output coordinates")
        self.__calibrated = True

    def transform_input_to_output(self, input):
        if not self.__calibrated:
            raise Exception("Transformation needs to be calibrated first")

    def transform_output_to_input(self, output):
        if not self.__calibrated:
            raise Exception("Transformation needs to be calibrated first")

    def model():
        a = 10

    def flatten_screen_coordinates(screen_coordinates_as_matrix):
        return screen_coordinates_as_matrix.flatten()

    def fit():
        flattened_projection = lambda *args : flatten_screen_coordinates(project_to_screen_with_perspective_multi(*args)) 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        p, cov = scipy.optimize.curve_fit(flattened_projection, real_world, flattened_screen, initial_guess)

