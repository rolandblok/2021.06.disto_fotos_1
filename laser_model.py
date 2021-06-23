import numpy as np
import math as math
import scipy.optimize

class ModelParams:
    def __init__(self):
        self.pix_x_2_phi_x = math.pi / (8*400) # guestimate , rad/pixel
        self.pix_y_2_phi_y = math.pi / (8*400) # guestimate , rad/pixel
        self.d = 2    # guestimate , meter
        self.e = 0.02 # guestimate , meter
    def _create_model_params(self, *args):
        self.e = args[0]
        self.d = args[1]
        self.pix_x_2_phi_x = args[2]
        self.pix_y_2_phi_y = args[3]

class LaserModel:

    def __init__(self):
        self.__calibrated = False
        self.mp = ModelParams()


    # model to project laser coordinates to screen meters
    def _project(self, las_pix_x, las_pix_y, model):
        phi_Y = las_pix_y * model.pix_x_2_phi_x
        phi_X = las_pix_x * model.pix_y_2_phi_y

        y = math.tan(phi_Y)
        x = (model.e + model.d / math.cos(phi_Y)) * math.tan(phi_X)



        return x,y

    def _project_perspective(self, x, y):



    def fit(self):
        flattened_projection = lambda *args : flatten_screen_coordinates(project_to_screen_with_perspective_multi(*args)) 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        p, cov = scipy.optimize.curve_fit(flattened_projection, real_world, flattened_screen, initial_guess)

