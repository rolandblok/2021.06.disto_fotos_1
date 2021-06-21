import numpy


# ===============
# fit the world coordinates (wx,wy) on the foto pixels (px,py)
# use least squares
# | px | = | A B | . | wx | + | ox |
# | py |   | C D |   | wy |   | oy |
#
#   P    =    M . w + O      
#
#   px = xMx.wx + xMy.wy + xMxy.wx*wy + ox
#   py = yMx.wx + yMy.wy + yMxy.wx*wy + oy 
#     etc
#
# | Px1 | = | 1 wx1 wy1  wx1*wy1  0   0   0     0    | . | ox   |
# | Py1 |   | 0  0   0      0     1  wx1 wy1 wx1*wy1 |   | oy   |
# | Px2 |   | 1 wx2 wy2  wx1*wy1  0   0   0     1    |   | xMx  |
# | Py2 |   | 0  0   0      0     1  wx2 wy2 wx1*wy1 |   | xMy  |
#   ...                                                  | xMxy |
#   ...                                                  | yMx  |
#   ...                                                  | yMy  |
#   ...                                                  | yMxy |
# etc
class TwoDToTwoDLinearTransformation:
    
    def __init__(self):
        self.__calibrated = False
        self.__matrix = None
        self.__offset = None

    def calibrate(self, input_coordinates, output_coordinates):
        if len(input_coordinates) != len(output_coordinates):
            raise Exception("Expect to get the same number of input and output coordinates")
        self.__calibrate_linear(input_coordinates, output_coordinates)
        self.__calibrated = True
    
    def __calibrate_linear(self, input_coordinates, output_coordinates):
        rows_for_x = [numpy.array([1.0, i[0], i[1], i[0]*i[1], 0.0,  0.0,  0.0,       0.0]) for i in input_coordinates]
        rows_for_y = [numpy.array([0.0, 0.0,   0.0,       0.0, 1.0, i[0], i[1], i[0]*i[1]]) for i in input_coordinates]
        vector_for_x = numpy.array([[o[0] for o in output_coordinates]]).T
        vector_for_y = numpy.array([[o[1] for o in output_coordinates]]).T

        design = numpy.vstack(rows_for_x + rows_for_y)
        data = numpy.vstack([vector_for_x, vector_for_y])

        least_squares_output = numpy.linalg.lstsq(design, data, rcond=None)
        parameters = least_squares_output[0]

        self.__matrix = parameters[0:4,0].reshape((2,2))
        self.__offset = parameters[4:,0].reshape((2,1))

    def transform_input_to_output(self, input_coordinate):
        if not self.__calibrated:
            raise Exception("Transformation needs to be calibrated first")
        input_as_array = numpy.array([[input_coordinate[0]], [input_coordinate[1]]])
        output_as_array = (self.__matrix @ input_as_array) + self.__offset
        return (output_as_array[0,0], output_as_array[1,0])

    def transform_output_to_input(self, output_coordinate):
        if not self.__calibrated:
            raise Exception("Transformation needs to be calibrated first")
        output_as_array = numpy.array([[output_coordinate[0]], [output_coordinate[1]]])
        input_as_array = numpy.linalg.inv(self.__matrix) @ (output_as_array - self.__offset)
        return (input_as_array[0,0], input_as_array[1,0])
