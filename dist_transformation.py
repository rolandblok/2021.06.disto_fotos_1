
import numpy
import math
import scipy.optimize


class DistTransformation:
    # More details on https://math.stackexchange.com/questions/296794/finding-the-transform-matrix-from-4-projected-points-with-javascript/339033#339033
    
    def __init__(self):
        #  Y = b * tan(s*las_sp_y - a)
        #  X = (d/cos(s*las_sp_y) + e) * tan(t*las_sp_x - c)

        self.s = math.pi / (8*400) # guestimate , rad/pixel
        self.t = math.pi / (8*400) # guestimate , rad/pixel
        self.d = 2    # guestimate , pixel 
        self.e = 0.02 # guestimate , pixel
        self.b = 1
        self.a = 0
        self.c = 0

        self.__calibrated = False

    @property
    def is_calibrated(self):
        return self.__calibrated

    def calibrate(self, laser_setpoints, distorted_laser_setpoints):
        if (len(laser_setpoints) != len(distorted_laser_setpoints)):
            raise Exception("Need exactly equal pairs to calibrate")
        self._calibrate(laser_setpoints, distorted_laser_setpoints)
        self.__calibrated = True

    # laser sp --> 
    def remove_disto(self, input_coordinate):
        if not self.__calibrated:
            raise Exception("Transformation needs to be calibrated first")
        x = input_coordinate[0]
        y = input_coordinate[1]
        las_sp = [0,0]
        las_sp[0] = (math.atan(y/self.b) + self.a ) / self.s
        las_sp[1] = (math.atan(x/(self.d/math.cos(self.t*y) + self.e)) + self.c) / self.t
        return las_sp


    def add_disto(self, input_coordinate ):
        if not self.__calibrated:
            raise Exception("Transformation needs to be calibrated first")
        self._add_disto_x(input_coordinate[0], input_coordinate[1], self.a,self.b,self.c,self.d,self.e,self.s,self.t)
        

    def _add_disto_x(self, las_sp_x, las_sp_y, a,b,c,d,e,s,t):
         X =  (d/math.cos(s*las_sp_y) + e) * math.tan(t*las_sp_x - c)
         return X
         
    def _add_disto_y(self, las_sp_y, a,b,c,d,e,s,t):
         Y = b * math.tan(s*las_sp_y - a)
         return Y

    def _add_disto(self, coordinates, a,b,c,d,e,s,t):
        # print(a,b,c,d,e,s,t)
        
        result = [[self._add_disto_x(x,y, a,b,c,d,e,s,t), self._add_disto_y(y, a,b,c,d,e,s,t)] for x,y in coordinates]
        stacked = numpy.vstack(result).flatten()
        return stacked


    def _calibrate(self, laser_setpoints, distorted_laser_setpoints):
        # flattened_projection = lambda *args : flatten_screen_coordinates(project_to_screen_with_perspective_multi(*args)) 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        laser_setpoints = numpy.vstack(laser_setpoints)
        distorted_laser_setpoints = numpy.vstack(distorted_laser_setpoints).flatten()
        p, cov = scipy.optimize.curve_fit(self._add_disto, laser_setpoints, distorted_laser_setpoints, [self.a,self.b,self.c,self.d,self.e,self.s,self.t])
        print(p)
        self.a,self.b,self.c,self.d,self.e,self.s,self.t = p
