import numpy as np
import math as math
import scipy.optimize

class ModelParams:
    def __init__(self):
        self.PARAMETER_LIST = ["mxx", "myx", "mzx", "mxy", "myy", "mzy", "mxz", "myz", "mzz", "tx", "ty", "tz"]
        self.M = [1]*12  # initial_guess
        # initial_guess = [1]*8

    def create_model_matrix(self, *args):
        # self.M = np.matrix([[args[0], 0, args[1], 0], [args[2], 0, args[3], 0], \
        #     [args[4], 0, args[5], 0], [args[5], 0, args[7],  0]])
        self.M = np.matrix([[args[0], args[1], args[2], 0], [args[3], args[4], args[5], 0], \
            [args[6], args[7], args[8], 0], [args[9], args[10], args[11], 0]])
        return self.M

    def get_matrix_as_parameters(self):
        m = self.M
        return m[0,0], m[0,1], m[0,2], m[1,0], m[1,1], m[1,2], m[2,0], m[2,1], m[2,2], m[3,0], m[3,1], m[3,2]



class PerspectiveModel:

    def __init__(self):
        self.__calibrated = False
        self.mp = ModelParams()
        self.dum_z = 2 # this parameters is finicky : 1, 0 or negative is nono
        self.perspective_matrix = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -1], [0, 0, 0, 0]])


    def project_to_screen_with_perspective_multi(self, coordinates, *args):
        M = self.mp.create_model_matrix(*args)
        result = [self.project_perspective(x,y,z, M) for x,y,z in coordinates]
        # result = [self.project_perspective(x,y,z, M, False) for x,y,z in coordinates] # WHY DOES THIS NOT WORK!>!>!>!>!?!?!?!>?!?>!!@#!@#@#$
        stacked = np.vstack(result)
        return stacked


    # ====
    # project a world coordinate towards a screen pixel coordinate
    def project_perspective(self, x, y, z=None, M=[], include_perspective=True):
        if len(M) == 0:
            M = self.mp.M
        if z == None:
            z = self.dum_z
        camera_coordinates =  np.matrix([x, y, z, 1]) @ M
        screen_coordinates = camera_coordinates @ self.perspective_matrix
        screen_coordinates_as_array = np.asarray(screen_coordinates).reshape(-1)
        if include_perspective:
            screen_coordinates_as_array = screen_coordinates_as_array / screen_coordinates_as_array[3]
        coordinates_xy_only = screen_coordinates_as_array[:2]
        return coordinates_xy_only.tolist()


    def fit(self, real_world_xy, screen_xy):
        real_world_xyz = [[X[0], X[1], self.dum_z] for X in real_world_xy]
        screen_xy = np.array(screen_xy)
        flattened_screen = screen_xy.flatten()

        print("real_world_xyz   " + str(real_world_xyz))
        print("flattened_screen " + str(flattened_screen))

        flattened_projection = lambda *args : self.project_to_screen_with_perspective_multi(*args).flatten()

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        p, cov = scipy.optimize.curve_fit(flattened_projection, real_world_xyz, flattened_screen, self.mp.M)
        self.mp.create_model_matrix(*p)
        print("scipy.curvefit model matrix " + str(self.mp.M))

