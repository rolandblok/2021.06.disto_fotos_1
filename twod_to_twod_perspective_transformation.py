
import numpy

class TwoDToTwoDPerspectiveTransformation:
    # More details on https://math.stackexchange.com/questions/296794/finding-the-transform-matrix-from-4-projected-points-with-javascript/339033#339033
    
    def __init__(self):
        self.__calibrated = False
        self.__matrix = None
        self.__matrix_inv = None

    @property
    def is_calibrated(self):
        return self.__calibrated

    def calibrate(self, input_coordinates, output_coordinates):
        print(f"{input_coordinates=}")
        print(f"{output_coordinates=}")
        if (len(input_coordinates) != 4) or (len(output_coordinates) != 4):
            raise Exception("Need exactly four coordinate pairs to calibrate")
        self._calibrate(input_coordinates, output_coordinates)
        self.__calibrated = True

    def transform_input_to_output(self, input_coordinate):
        if not self.__calibrated:
            raise Exception("Transformation needs to be calibrated first")
        return self._transform(self.__matrix, input_coordinate)
        
    def transform_output_to_input(self, output_coordinate):
        if not self.__calibrated:
            raise Exception("Transformation needs to be calibrated first")
        return self._transform(self.__matrix_inv, output_coordinate)

    def _transform(self, matrix, coordinate):
        coord_as_array = numpy.array([[coordinate[0]], [coordinate[1]], [1.0]])
        result_raw = matrix @ coord_as_array
        result_raw = result_raw / result_raw[2,0]
        return (result_raw[0,0], result_raw[1,0])

    def _calibrate(self, input_coordinates, output_coordinates):
        A = self._homogenize_to_matrix(input_coordinates)
        Ainv = numpy.linalg.inv(A)
        B = self._homogenize_to_matrix(output_coordinates)
        C = B @ Ainv
        self.__matrix = C
        self.__matrix_inv = numpy.linalg.inv(C)
    
    def _homogenize_to_matrix(self, coordinates):
        if len(coordinates) != 4:
            raise Exception("Need exactly four coordinates to homogenize")
        m = numpy.array([[coordinates[0][0], coordinates[1][0], coordinates[2][0]],
                         [coordinates[0][1], coordinates[1][1], coordinates[2][1]],
                         [1.0, 1.0, 1.0]])
        v = numpy.array([[coordinates[3][0]], [coordinates[3][1]], [1.0]])
        all_outs = numpy.linalg.lstsq(m, v, rcond="warn")
        x = all_outs[0]
        return m * x.T

