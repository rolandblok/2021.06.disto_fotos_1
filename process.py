import math
import numpy as np
print("numpy version  : " + np.version.version)
import matplotlib.pyplot as plot
from json import JSONEncoder
import cv2 as cv2
print("opencv version : " + cv2.__version__ )
import json
import argparse
import os.path
from math import *

from twod_to_twod_perspective_transformation import TwoDToTwoDPerspectiveTransformation
from dist_transformation import DistTransformation


# ===================================================
# what do/should we do : map two coordinate systems
#  laser_pixel (las_pix) ===> world_meters (wor_met) ===> photo_pixels (pho_pix)
#
#  1st : manual create reference world coordinates (world --> photo pixels)
#    fit the world meters to photo pixels 
#  2nd : find the laser spots via image processing
#    map those to laser pixel grid.
#  3rd : convert laser-photo pixels --> world coordinates
#
#  4th : fit laser pixel grid towards world coordinates.

class LaserFoto:
    LASER_A= "laser model basic"
    LASER_B= "laser model sticker"
    
    PROJECT_STRAIT="straight"
    PROJECT_ANGLE="angle"

    def __init__(self, file_name, start, step, mid, lines, laser_model, laser_axis, photo_axis) : 
        self.filename = file_name
        self.json_file_name = file_name + ".json"
        self.start = start
        self.step = step
        self.mid = mid
        self.no_laser_spots_per_line = lines
        self.laser_model = laser_model
        self.laser_axis = laser_axis
        self.photo_axis = photo_axis
    def __str__(self):
        return f"file: {self.filename} laser_model: {self.laser_model}"


laser_photos = []                                                                           # laser projection        photo projection
laser_photos.append(LaserFoto("fotos_2/20210618_095352.JPG", 50, 50, 400, 15, LaserFoto.LASER_A, LaserFoto.PROJECT_STRAIT, LaserFoto.PROJECT_ANGLE ))
laser_photos.append(LaserFoto("fotos_2/20210618_095410.JPG", 50, 50, 400, 15, LaserFoto.LASER_A, LaserFoto.PROJECT_STRAIT, LaserFoto.PROJECT_ANGLE ))

laser_photos.append(LaserFoto("fotos_2/20210618_095516.JPG", 50, 50, 400, 15, LaserFoto.LASER_A, LaserFoto.PROJECT_ANGLE, LaserFoto.PROJECT_STRAIT ))
laser_photos.append(LaserFoto("fotos_2/20210618_095534.JPG", 50, 50, 400, 15, LaserFoto.LASER_A, LaserFoto.PROJECT_ANGLE, LaserFoto.PROJECT_STRAIT ))
laser_photos.append(LaserFoto("fotos_2/20210618_095609.JPG", 50, 50, 400, 15, LaserFoto.LASER_A, LaserFoto.PROJECT_ANGLE, LaserFoto.PROJECT_STRAIT ))
laser_photos.append(LaserFoto("fotos_2/20210618_095618.JPG", 50, 50, 400, 15, LaserFoto.LASER_A, LaserFoto.PROJECT_ANGLE, LaserFoto.PROJECT_STRAIT ))

laser_photos.append(LaserFoto("fotos_2/20210618_100631.JPG", 50, 50, 400, 15, LaserFoto.LASER_B, LaserFoto.PROJECT_ANGLE, LaserFoto.PROJECT_STRAIT ))
laser_photos.append(LaserFoto("fotos_2/20210618_100642.JPG", 50, 50, 400, 15, LaserFoto.LASER_B, LaserFoto.PROJECT_ANGLE, LaserFoto.PROJECT_STRAIT ))

laser_photos.append(LaserFoto("fotos_2/20210618_101129.JPG", 50,100, 400,  8, LaserFoto.LASER_B, LaserFoto.PROJECT_ANGLE, LaserFoto.PROJECT_STRAIT ))
laser_photos.append(LaserFoto("fotos_2/20210618_101220.JPG", 50,100, 400,  8, LaserFoto.LASER_B, LaserFoto.PROJECT_ANGLE, LaserFoto.PROJECT_STRAIT ))
laser_photos.append(LaserFoto("fotos_2/20210618_101230.JPG", 50,100, 400,  8, LaserFoto.LASER_B, LaserFoto.PROJECT_ANGLE, LaserFoto.PROJECT_STRAIT ))

laser_photos.append(LaserFoto("fotos_2/20210618_101335.JPG", 50,100, 400,  8, LaserFoto.LASER_B, LaserFoto.PROJECT_STRAIT, LaserFoto.PROJECT_ANGLE ))
laser_photos.append(LaserFoto("fotos_2/20210618_101359.JPG", 50,100, 400,  8, LaserFoto.LASER_B, LaserFoto.PROJECT_STRAIT, LaserFoto.PROJECT_ANGLE ))

laser_photos.append(LaserFoto("fotos_2/20210618_101612.JPG", 50, 50, 400, 15, LaserFoto.LASER_B, LaserFoto.PROJECT_STRAIT, LaserFoto.PROJECT_ANGLE ))
laser_photos.append(LaserFoto("fotos_2/20210618_101622.JPG", 50, 50, 400, 15, LaserFoto.LASER_B, LaserFoto.PROJECT_STRAIT, LaserFoto.PROJECT_ANGLE ))

laser_photos.append(LaserFoto("fotos_2/20210618_102537.JPG", 50,100, 400,  8, LaserFoto.LASER_A, LaserFoto.PROJECT_STRAIT, LaserFoto.PROJECT_ANGLE ))
laser_photos.append(LaserFoto("fotos_2/20210618_102559.JPG", 50,100, 400,  8, LaserFoto.LASER_A, LaserFoto.PROJECT_STRAIT, LaserFoto.PROJECT_ANGLE ))

# not usable, no complete view of spots :-(
# laser_photos.append(LaserFoto("fotos_2/20210618_102725.JPG", 50,100, 400,  8, LaserFoto.LASER_A, LaserFoto.PROJECT_ANGLE, LaserFoto.PROJECT_STRAIT ))
# laser_photos.append(LaserFoto("fotos_2/20210618_102736.JPG", 50,100, 400,  8, LaserFoto.LASER_A, LaserFoto.PROJECT_ANGLE, LaserFoto.PROJECT_STRAIT ))

active_laser_photo = 0

# =====================
# Create the references
ref_wor_met = [[-1, 1], [0, 1], [1,1], # wereld coordinaten in meters.
               [-1, 0], [0, 0], [1,0], 
               [-1,-1], [0,-1],[1,-1]]
laser_las_pix = []
def initialize_laser_las_pix(laser_photo):
    global laser_las_pix
    loadJson(laser_photo.json_file_name)
    laser_las_pix.clear()
    for y_i in range(0,laser_photo.no_laser_spots_per_line) :
        for x_i in range(0,laser_photo.no_laser_spots_per_line) :
            laser_las_pix.append([laser_photo.start + x_i*laser_photo.step - laser_photo.mid, laser_photo.start + y_i*laser_photo.step - laser_photo.mid])
    
# ================
# create the other globals
laser_pho_pix = []      # laser spots in photo pixels  : determined by openCV and store in JSON
ref_pho_pix   = []      # reference in photo pixels    : determined by click on photo and store in JSON
laser_pho_pix_from_reference = []      # check the references through the fitted model : used for photo plot

wor_to_pho_perspective_model = TwoDToTwoDPerspectiveTransformation()
las_to_wor_perspective_model = TwoDToTwoDPerspectiveTransformation()
disto_transform_model        = DistTransformation()

DRAW_CIRCLE_RADIUS = 14
DRAW_REF_RADIUS    = 14
scale = 0.25
window_name = "laser photo"
# =====================
# json serialize helper
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# =====================================
# load json file.
def loadJson(json_file_name):
    global ref_pho_pix, laser_pho_pix
    if (os.path.isfile(json_file_name)):
        print(f"Loading {json_file_name}")
        with open(json_file_name, 'r', encoding='utf-8') as json_file:
            json_data = json_file.read()
        json_data = json.loads(json_data)
        ref_pho_pix = json_data["ref_pho_pix"]
        laser_pho_pix = json_data["laser_pho_pix"]
    else:
        ref_pho_pix = []
        laser_pho_pix = []


# =====================================
# save json file.
def saveJson(json_file_name):
    with open(json_file_name, 'w', encoding='utf-8') as json_file:
        json_data = {}
        json_data["ref_pho_pix"] = ref_pho_pix
        json_data["laser_pho_pix"] = laser_pho_pix
        json.dump(json_data, json_file, ensure_ascii=False, indent=4, cls=NumpyArrayEncoder)

    if disto_transform_model.is_calibrated :
        disto_model_file_name = "disto_model_params.txt"
        with open(os.getcwd()+disto_model_file_name+".txt", 'a', encoding='utf-8') as data_file:

            for param in disto_transform_model.params :
                data_file.write(str(param) + ', ')
            data_file.write('\n ')


# =====================================
# display the image.
def image_show():

    img = img_org.copy()
    
    cv2.putText(img, laser_photos[active_laser_photo].filename, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 3 , color=(0,0,55), thickness = 2)

    counter = 0
    for center in laser_pho_pix:
        cv2.circle(img, (int(center[0]), int(center[1])), int(DRAW_CIRCLE_RADIUS), color=(0, 0, 255), thickness=2) # (B, G, R)
        cv2.putText(img, str(counter) ,(center[0]+4,center[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 , color=(0,0,55), thickness = 1)
        counter += 1
    
    counter = 0
    for r in ref_pho_pix:
        counter += 1
        RR = int(DRAW_REF_RADIUS/2)
        cv2.drawMarker(img, (r[0],r[1]), color=(0,255,55),thickness=1 ) 
        cv2.putText(img, str(counter) ,(r[0]+4,r[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 5 , color=(0,255,55), thickness = 2)

    if wor_to_pho_perspective_model.is_calibrated:
        for w,p in zip(ref_wor_met, ref_pho_pix):
            photo_coor = wor_to_pho_perspective_model.transform_input_to_output((w[0], w[1]))
            cv2.circle(img, (int(photo_coor[0]), int(photo_coor[1])), int(DRAW_CIRCLE_RADIUS+4), color=(255, 255, 255), thickness=2) # (B, G, R)
            wereld = wor_to_pho_perspective_model.transform_output_to_input((p[0], p[1]))

    counter = 0
    for center in laser_pho_pix_from_reference:
        cv2.circle(img, (int(center[0]), int(center[1])), int(DRAW_CIRCLE_RADIUS+4), color=(255, 0, 255), thickness=2) # (B, G, R)
        counter += 1
    

        
    img = img[crop[1][0]:crop[1][1], crop[0][0]:crop[0][1]] #https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python#15589825
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    cv2.imshow(window_name, img)

# =======================
# IMAGE PROCESS THE LASERS
def image_process_laser_dots(image):
    # https://stackoverflow.com/questions/51846933/finding-bright-spots-in-a-image-using-opencv#51848512
        #  constants
    BINARY_THRESHOLD = 80
    CONNECTIVITY = 10
    #  convert to gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #  extract edges
    binary_image = cv2.Laplacian(gray_image, cv2.CV_8UC1)
    #  fill in the holes between edges with dilation
    dilated_image = cv2.dilate(binary_image, np.ones((15, 15)))
    # cv2.imshow("dilated_image", dilated_image)
    #  threshold the black/ non-black areas
    _, thresh = cv2.threshold(dilated_image, BINARY_THRESHOLD, 200, cv2.THRESH_BINARY)
    #  find connected components
    components = cv2.connectedComponentsWithStats(thresh, CONNECTIVITY, cv2.CV_32S)
    #  draw circles around center of components
    #see connectedComponentsWithStats function for attributes of components variable
    global laser_pho_pix
    laser_pho_pix = []
    components3 = components[3]
    for component in components3:
        # transform to serializable data
        laser_pho_pix.append((int(component[0]), int(component[1])))

    laser_map_and_sort(laser_photos[active_laser_photo])
    image_show()

# =============
# sort the references (asume it's nine of them (3X3)) and fit them to the meters
def x_value(v) : return v[0]
def y_value(v) : return v[1]
def sort_references():
    global ref_pho_pix
    if(len(ref_pho_pix) != 9) :
        print("cannot sort reference, only when 9")
        return
    ref_pho_pix = sorted(ref_pho_pix, key=y_value)
    for i in [0,1,2]:
        # ref_pho_pix[3*i:(3*i)+3].sort(key=x_value)
        ref_pho_pix[3*i:(3*i)+3] = sorted(ref_pho_pix[3*i:(3*i)+3], key=x_value)




# =============
# get the corners of the list : return the points and the indices
def det_matrix_corners(M):
    # find the corners
    x_plus_y = [c[0]+c[1] for c in M]
    x_min_y  = [c[0]-c[1] for c in M]
    
    min_value = min(x_plus_y)
    min_index_x_plus_y = x_plus_y.index(min_value)
    min_value = min(x_min_y)
    min_index_x_min_y = x_min_y.index(min_value)

    max_value = max(x_plus_y)
    max_index_x_plus_y = x_plus_y.index(max_value)
    max_value = max(x_min_y)
    max_index_x_min_y = x_min_y.index(max_value)

    corners   = np.zeros((2,2,2))
    corners[0][0] =  M[min_index_x_plus_y]  # top left
    corners[1][0] =  M[max_index_x_min_y]   # top right
    corners[0][1] =  M[min_index_x_min_y]   # bot left
    corners[1][1] =  M[max_index_x_plus_y]  # bot right
    corners_i = np.zeros((2,2))
    corners_i[0][0] =  min_index_x_plus_y  # top left
    corners_i[1][0] =  max_index_x_min_y   # top right
    corners_i[0][1] =  min_index_x_min_y   # bot left
    corners_i[1][1] =  max_index_x_plus_y  # bot right
    
    return corners, corners_i.astype(np.int32)

# =============
# helper : closest point : return index and distance
def closest_node(point, points):
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    i = np.argmin(dist_2)
    d = sqrt(dist_2[i])
    return i, d
# ==============
# helper : return if point p is above line created from points a and b
def isabove(p, a,b): 
    return (np.cross(p-a, b-a) > 0) # https://stackoverflow.com/questions/45766534/finding-cross-product-to-find-points-above-below-a-line-in-matplotlib


# =============
# map the lasers to the set setoints, and sort them
def laser_map_and_sort(laser_photo):
    global laser_pho_pix

    if(len(ref_pho_pix) != 9) :
        print("cannot sort reference, only when 9")
        return
    no_expected_laser_spots = laser_photo.no_laser_spots_per_line * laser_photo.no_laser_spots_per_line
    if (no_expected_laser_spots != len(laser_pho_pix)):
        print("Warning : no expected laser spots not equal to no laser spots")
        print(" expected " + str(no_expected_laser_spots))
        print(" actual   " + str(len(laser_pho_pix)))
        return

    # create the result arrayss
    laser_pho_pix_work = laser_pho_pix.copy()
    laser_centers_sort = []
    counter = 0

    # find closest distance:
    p = laser_pho_pix_work[0]
    laser_pho_pix_work.pop(0)
    c, min_dist = closest_node(p, laser_pho_pix_work)
    min_dist = int(round(min_dist))
    
    # repair work list
    laser_pho_pix_work = laser_pho_pix.copy()

    # for each horizontal line, get the points, sort them and store them
    for line in range(0, laser_photo.no_laser_spots_per_line):
        laser_corners, laser_corners_i = det_matrix_corners(laser_pho_pix_work)
        A = laser_corners[0][0]
        laser_pho_pix_corner_work = laser_pho_pix_work.copy()
        
        laser_pho_pix_corner_work.pop(laser_corners_i[0][0])
        c, a_min_dist = closest_node(A, laser_pho_pix_corner_work)

        laser_corners, laser_corners_i = det_matrix_corners(laser_pho_pix_work)
        B = laser_corners[1][0]
        laser_pho_pix_corner_work = laser_pho_pix_work.copy()
        laser_pho_pix_corner_work.pop(laser_corners_i[1][0])
        c, b_min_dist = closest_node(B, laser_pho_pix_corner_work)


        # create laser estimates line A, from the top corners
        A[1] += a_min_dist/2 # offset the y point with 50 pixels
        B[1] += b_min_dist/2 # offset the y point with 50 pixels
        laser_centers_above_line = []
        laser_centers_below_line = []
        # get all point above the line A-B, remove them from work list, and sort them.
        for s_i,s in enumerate(laser_pho_pix_work):
            if isabove(s, A,B) :
                laser_centers_above_line.append(s)
            else:
                laser_centers_below_line.append(s)
        laser_pho_pix_work = laser_centers_below_line
        laser_centers_above_line = sorted(laser_centers_above_line, key=x_value )
        # add the sorted points the the total list.
        for lci in laser_centers_above_line:  # https://www.askpython.com/python/two-dimensional-array-in-python
            laser_centers_sort.append(lci)

    laser_pho_pix = laser_centers_sort

    return 

# ============================
# model one by one all projections.
def model_projections(laser_photo, plot_on=True):
    global disto_transform_model
    no_expected_laser_spots = laser_photo.no_laser_spots_per_line * laser_photo.no_laser_spots_per_line
    if (no_expected_laser_spots != len(laser_pho_pix)):
        print("Warning : no expected laser spots not equal to no laser spots")
        print(" expected " + str(no_expected_laser_spots))
        print(" actual   " + str(len(laser_pho_pix)))
        return

    # ..............
    # project reference to photo pixels : pespective removal possible
    corners = [0, 2, 6, 8]
    inputs = [ref_wor_met[i] for i in corners]
    outputs = [ref_pho_pix[i] for i in corners]
    wor_to_pho_perspective_model.calibrate(inputs, outputs)
    # TODO : residuals of fit
    # print ("2D perspex fit inverse" +  str(wereld[0]-w[0]) + " --> " + str(wereld[1]-w[1]))

    laser_wor_met       = []      # laser sports in world meters : determined by reference fit         global laser_sp_perp_free
    laser_sp_perp_free  = []      # laser spot from photo to laser setpoints : remove photo and laser perspective

    # ..............
    # use corners of lasers to determine laser perspective
    # determine laser corners
    laser_corners_pho_pix, laser_corners_pho_pix_i = det_matrix_corners(laser_pho_pix)
    laser_corners_pho_pix_i = laser_corners_pho_pix_i.flatten()
    # remove photo perspective from corners laser-photo-pixels
    laser_corners_wor_met = [wor_to_pho_perspective_model.transform_output_to_input(laser_pho_pix[i]) for i in laser_corners_pho_pix_i]
    laser_corners_las_pix = [laser_las_pix[i] for i in laser_corners_pho_pix_i]
    # fit the laser perspective on the corners
    las_to_wor_perspective_model.calibrate(laser_corners_las_pix, laser_corners_wor_met)

    # remove photo and laser perspective 
    for lpp in laser_pho_pix:
        # remove photo perspective (for plot only)
        lwm = wor_to_pho_perspective_model.transform_output_to_input(lpp)
        laser_wor_met.append(lwm)

        # remove laser perspective also
        lsp_d = las_to_wor_perspective_model.transform_output_to_input(lwm)
        laser_sp_perp_free.append(lsp_d)

    # plot the laser spots with both perspectives removed.
    if (plot_on):
        fig1 = plot.figure()
        x = [c[0] for c in laser_wor_met]
        y = [c[1] for c in laser_wor_met]
        plot.plot(x,y,'.')
        plot.title("laser spots with photo perspective removed")

        fig2 = plot.figure()
        x = [c[0] for c in laser_sp_perp_free]
        y = [c[1] for c in laser_sp_perp_free]
        plot.plot(x,y,'x', label="laser perspective removed")
        plot.title("laser spots with both photo and laser perspective removed")


    # calibrate the distortion of the laser
    laser_sp_persp_n_disto_free  = []      # laser spot photo to setpoints : removed photo persp, laser persp, laser disto.

    disto_transform_model.calibrate(laser_las_pix , laser_sp_perp_free)

    laser_sp_persp_n_disto_free = [disto_transform_model.remove_disto(l) for l in laser_sp_perp_free]
    
    if (plot_on):
        x = [c[0] for c in laser_sp_persp_n_disto_free]
        y = [c[1] for c in laser_sp_persp_n_disto_free]
        plot.plot(x,y, 'o', label="disto also removed")
        rx = [c[0] for c in laser_las_pix]
        ry = [c[1] for c in laser_las_pix]
        plot.plot(rx,ry, '.', color="red", label="laser setpoints")
        plot.title("laser spots with all removed removed")
        plot.legend()

    # for checking :
    global laser_pho_pix_from_reference
    laser_pho_pix_from_reference = [wor_to_pho_perspective_model.transform_input_to_output( \
                                      las_to_wor_perspective_model.transform_input_to_output( \
                                           disto_transform_model.add_disto(r))) for r in laser_las_pix]


    distance = lambda a, b: math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    distances = [distance(a,b) for a, b in zip(laser_pho_pix, laser_pho_pix_from_reference)]
    max_distance = max(distances)
    print(f"{max_distance=}")

    if plot_on:
        plot.ion()
        plot.show()
        image_show()

    return max_distance




# ===============
# MOUSE HANDLING
l_mouse_down = False
r_mouse_down = False
r_mouse_down_last = [0,0]
def mouse_call(event, x, y, flags, param):
    global l_mouse_down, r_mouse_down
    global scale, img_height, img_width, crop, r_mouse_down_last, laser_pho_pix, ref_pho_pix
    if event == cv2.EVENT_MOUSEMOVE:
        if(r_mouse_down):
            crop[0][0] += (x - r_mouse_down_last[0] )
            if (crop[0][0] < 0) : 
                crop[0][0] = 0
            crop[1][0] += (y - r_mouse_down_last[1] )
            if (crop[1][0] < 0) : 
                crop[1][0] = 0
            r_mouse_down_last = [x,y]
        return
    elif event == cv2.EVENT_LBUTTONDOWN:
        l_mouse_down = True
        # print(" l mouse  " + str(x) + " " + str(y))
        x += crop[0][0]
        y += crop[1][0]
        if (scale == 1):
            if (flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON) :
                # REMOVE UNWANTED LASER OR REFERENCES
                print("CNTRL")
                
                cntr_cnt = 0
                for center in laser_pho_pix:
                    # print(str(center) + str((x,y)))
                    if ((abs(x - int(center[0])) < DRAW_CIRCLE_RADIUS) and 
                        (abs(y - int(center[1])) < DRAW_CIRCLE_RADIUS)):
                        print ("HIT laser " + str(cntr_cnt))
                        laser_pho_pix.pop(cntr_cnt)
                        laser_map_and_sort(laser_photos[active_laser_photo])
                        break
                    cntr_cnt += 1

                cntr_cnt = 0
                for center in ref_pho_pix:
                    # print(str(center) + str((x,y)))
                    if ((abs(x - int(center[0])) < DRAW_CIRCLE_RADIUS) and 
                        (abs(y - int(center[1])) < DRAW_CIRCLE_RADIUS)):
                        print ("HIT reference marker")
                        ref_pho_pix.pop(cntr_cnt)
                        break
                    cntr_cnt += 1
            elif (flags == cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_FLAG_LBUTTON) :
                # add UNWANTED LASER OR REFERENCES
                print("alt")
                laser_pho_pix.append((x,y))
                laser_map_and_sort(laser_photos[active_laser_photo])
            else:
                # ADD REFERENCE
                if (len(ref_pho_pix) < 9):
                    ref_pho_pix.append((x,y))
                    sort_references()
                else :
                    print("too many reference marks already, first remove some ")


    elif (event == cv2.EVENT_LBUTTONUP):
        l_mouse_down = False
        # print(" l button " + str(x))

    elif (event == cv2.EVENT_RBUTTONDOWN):
        r_mouse_down = True
        r_mouse_down_last = [x,y]
        # print(" r mouse  " + str(x) + " " + str(y))
    elif (event == cv2.EVENT_RBUTTONUP):
        r_mouse_down = False
        # print(" r mouse  " + str(x) + " " + str(y))
        
    elif (event == cv2.EVENT_MOUSEWHEEL) :
        # print(" m wheel  " + str(x) + " " + str(y) + " " + str(flags) + " " + str(param))
        if (flags > 0):
            scale = scale * 2 
            if scale > 1:
                scale = 1
        else :
            scale = scale / 2 
            if scale < 0.24 :
                scale = 0.25
        print ("scale " + str(scale))

    image_show()


img_width  = 0
img_height = 0
img_org    = 0
def load_image(laser_photo):
    global img_height, img_width, crop, img_org
    img_org = cv2.imread(laser_photo.filename, cv2.IMREAD_COLOR)
    img_height = img_org.shape[0]
    img_width  = img_org.shape[1]
    crop       = [[0, img_width], [0, img_height]]   # [[x1,x2], [y1,y2]]
    image_show()



# ///////////
# main script
# ///////////

if __name__ == "__main__":


    initialize_laser_las_pix(laser_photos[active_laser_photo])
    load_image(laser_photos[active_laser_photo])
    cv2.setMouseCallback(window_name, mouse_call)

    print("press q to QUIT")
    print("      r to RESET VIEW")
    print("      l to LASER spot image process")
    print("      c to clear Laser MODEL")
    print("      S to clear Laser map and sort")
    print("      m to perspective fit the reference photo-world correction model")
    print("      RMB to DRAG VIEW")
    print("      LMB to SET MARKER and/or laser spot")
    print("      CNTRL-LMB to REMOVE LASER and/or REFERENCE POINT (only @scale 1")

    run = True
    while(run) :
        plot.pause(0.01)

        key = cv2.waitKeyEx(30)
        # if key != -1 : print(".." + str(key) + "..")
        if (key == ord("q")) : 
            run = False
        elif (key == ord("r")):
            crop          = [[0, img_width], [0, img_height]]   # img[y:y+h, x:x+w]
            scale = 0.25
            image_show()
        elif (key == ord("l")):
            print( "MODELING LASERS")
            image_process_laser_dots(img_org)
            image_show()
        elif (key == ord("c")):
            print( "CLEAR LASERS")
            laser_pho_pix = []
            image_show()
        elif (key == ord("s")):
            laser_map_and_sort(laser_photos[active_laser_photo])
            image_show()
        elif (key == ord("m")):
            model_projections(laser_photos[active_laser_photo])

            image_show()
        elif (key == 2424832): # left arrow
            if active_laser_photo == 0 :
                active_laser_photo = len(laser_photos) - 1
            else : 
                active_laser_photo -= 1
            initialize_laser_las_pix(laser_photos[active_laser_photo])
            load_image(laser_photos[active_laser_photo])

        elif (key == 2555904): # right arrow
            if active_laser_photo == len(laser_photos) - 1 :
                active_laser_photo = 0
            else : 
                active_laser_photo += 1
            initialize_laser_las_pix(laser_photos[active_laser_photo])
            load_image(laser_photos[active_laser_photo])


                
            
    saveJson(laser_photos[active_laser_photo].json_file_name)
    cv2.destroyAllWindows()

