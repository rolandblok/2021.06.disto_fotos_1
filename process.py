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

from perspective_model import *


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



# filename = "fotos_2/20210618_095352.JPG"
# laser_las_pix_coor_start       = 50
# laser_las_pix_coor_step        = 50
# laser_las_pix_coor_mid_pixel   = 400
# no_laser_spots_per_line        = 15

# filename = "fotos_2/20210618_101129.JPG"
# laser_las_pix_coor_start         = 50
# laser_las_pix_coor_step          = 100
# laser_las_pix_coor_mid_pixel     = 400
# no_laser_spots_per_line          = 8

filename = "fotos_2/20210618_095410.JPG"
laser_las_pix_coor_start         = 50
laser_las_pix_coor_step          = 100
laser_las_pix_coor_mid_pixel     = 400
no_laser_spots_per_line          = 15


# =====================
# Create the references
ref_wor_met = [[-1, 1], [0, 1], [1,1], # wereld coordinaten in meters.
               [-1, 0], [0, 0], [1,0], 
               [-1,-1], [0,-1],[1,-1]]
laser_las_pix = []
for y_i in range(0,no_laser_spots_per_line) :
    for x_i in range(0,no_laser_spots_per_line) :
        laser_las_pix.append([laser_las_pix_coor_start + x_i*laser_las_pix_coor_step - laser_las_pix_coor_mid_pixel, laser_las_pix_coor_start + y_i*laser_las_pix_coor_step - laser_las_pix_coor_mid_pixel])


# ================
# create the other globals
json_file_name = filename + ".json"
img_org = cv2.imread(filename, cv2.IMREAD_COLOR)
img_height = img_org.shape[0]
img_width  = img_org.shape[1]
crop          = [[0, img_width], [0, img_height]]   # [[x1,x2], [y1,y2]]
laser_pho_pix = []      # laser spots in photo pixels  : determined by openCV
laser_wor_met = []      # laser sports in world meters : determined by reference fit 
ref_pho_pix   = []      # reference in photo pixels    : determined by click on photo

world_photo_fit_params = []
world_photo_perspective_model = 0

scale = 0.25
DRAW_CIRCLE_RADIUS = 14
DRAW_REF_RADIUS    = 14

# =====================
# json serialize helper
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# =====================================
# load json file.
def loadJson():
    global ref_pho_pix, laser_pho_pix
    if (os.path.isfile(json_file_name)):
        with open(json_file_name, 'r', encoding='utf-8') as json_file:
            json_data = json_file.read()
        json_data = json.loads(json_data)
        ref_pho_pix = json_data["ref_pho_pix"]
        laser_pho_pix = json_data["laser_pho_pix"]
        if (len(ref_pho_pix) == 9):
            fit_world_photo()


# =====================================
# save json file.
def saveJson():
    with open(json_file_name, 'w', encoding='utf-8') as json_file:
        json_data = {}
        json_data["ref_pho_pix"] = ref_pho_pix
        json_data["laser_pho_pix"] = laser_pho_pix
        json.dump(json_data, json_file, ensure_ascii=False, indent=4, cls=NumpyArrayEncoder)



# =====================================
# display the image.
def image_show():
    img = img_org.copy()
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

    if (world_photo_fit_params):
        for w in ref_wor_met:
            photo_coor = world_to_foto(w)
            print ("linear fit " + str(w) + " --> " + str(photo_coor))
            cv2.circle(img, (int(photo_coor[0]), int(photo_coor[1])), int(DRAW_CIRCLE_RADIUS), color=(255, 0, 0), thickness=2) # (B, G, R)

    if (world_photo_perspective_model):
        for w in ref_wor_met:
            photo_coor = world_photo_perspective_model.project_perspective(w[0], w[1])
            print ("perspex fit " +  str(w) + " --> " + str(photo_coor))
            cv2.circle(img, (int(photo_coor[0]), int(photo_coor[1])), int(DRAW_CIRCLE_RADIUS+2), color=(255, 255, 0), thickness=2) # (B, G, R)


        
    img = img[crop[1][0]:crop[1][1], crop[0][0]:crop[0][1]] #https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python#15589825
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    cv2.imshow(filename, img)

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
    _, thresh = cv2.threshold(dilated_image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
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
        print(str(i))
        # ref_pho_pix[3*i:(3*i)+3].sort(key=x_value)
        ref_pho_pix[3*i:(3*i)+3] = sorted(ref_pho_pix[3*i:(3*i)+3], key=x_value)
    fit_world_photo()




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
    corners_i = np.zeros((2,2,1))
    corners_i[0][0] =  min_index_x_plus_y  # top left
    corners_i[1][0] =  max_index_x_min_y   # top right
    corners_i[0][1] =  min_index_x_min_y   # bot left
    corners_i[1][1] =  max_index_x_plus_y  # bot right
    
    return corners, corners_i

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
# convert lasers photo pixels towards laser world meters (remove photo perspective)
def laser_pho_pix_2_laser_wor_met():
    global laser_wor_met
    laser_wor_met = []
    for lpp in laser_pho_pix:
        lwm = foto_to_world (lpp)
        laser_wor_met.append(lwm)
    return

# =============
# map and sort the lasers
def laser_map():
    global laser_pho_pix

    no_expected_laser_spots = no_laser_spots_per_line * no_laser_spots_per_line
    if (no_expected_laser_spots != len(laser_pho_pix)):
        print("Warning : no expected laser spots not equal to no laser spots")
        print(" expected " + str(no_expected_laser_spots))
        print(" actual   " + str(len(laser_pho_pix)))

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
    for line in range(0, no_laser_spots_per_line):
        laser_corners, laser_corners_i = det_matrix_corners(laser_pho_pix_work)
        # create laser estimates line A, from the top corners
        A = laser_corners[0][0]
        A[1] += min_dist/2 # offset the y point with 50 pixels
        B = laser_corners[1][0]
        B[1] += min_dist/2 # offset the y point with 50 pixels
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

# ===============
# fit the world coordinates (wx,wy) on the foto pixels (px,py)
# use least squares
# | px | = | ox | + | A B | . | wx | 
# | py |   | oy | + | C D |   | wy |   
#
#   P    =  O  +  M . w
#
#   px = ox + xMx.wx + xMy.wy + xMxy.wx*wy
#   py = oy + yMx.wx + yMy.wy + yMxy.wx*wy 
#     etc
#
# | Px1 | = | 1 wx1 wy1  wx1*wy1  0   0   0     0    | . | ox   |
# | Py1 |   | 0  0   0      0     1  wx1 wy1 wx1*wy1 |   | xMx  |
# | Px2 |   | 1 wx2 wy2  wx1*wy1  0   0   0     1    |   | xMy  |
# | Py2 |   | 0  0   0      0     1  wx2 wy2 wx1*wy1 |   | xMxy |
#   ...                                                  | oy   |
#   ...                                                  | yMx  |
#   ...                                                  | yMy  |
#   ...                                                  | yMxy |
# etc
def fit_world_photo():
    global world_photo_fit_params

    M = np.zeros((18,8))
    P = np.zeros((18,1))
    row_cnt = 0
    for p,w in zip(ref_pho_pix,ref_wor_met):
        P[row_cnt]   = p[0]
        P[row_cnt+1] = p[1]

        M[row_cnt,0] = 1
        M[row_cnt,1] = w[0]
        M[row_cnt,2] = w[1]
        M[row_cnt,3] = w[0]*w[1]
        M[row_cnt,4] = 0
        M[row_cnt,5] = 0
        M[row_cnt,6] = 0
        M[row_cnt,7] = 0

        M[row_cnt+1,0] = 0
        M[row_cnt+1,1] = 0
        M[row_cnt+1,2] = 0
        M[row_cnt+1,3] = 0
        M[row_cnt+1,4] = 1
        M[row_cnt+1,5] = w[0]
        M[row_cnt+1,6] = w[1]
        M[row_cnt+1,7] = w[0]*w[1]

        row_cnt += 2

    Ox, xMx, xMy, xMxy, Oy, yMx, yMy, yMxy  = np.linalg.lstsq(M, P, rcond=None)[0]
    world_photo_fit_params=(*Ox, *xMx, *xMy, *xMxy, *Oy, *yMx, *yMy, *yMxy)

    return

# ===============
# calculate the world coordinates towards photo pixel coordinates
def world_to_foto(point_wor_met):
    x = point_wor_met[0]
    y = point_wor_met[1]
    wp = world_photo_fit_params
    Px = wp[0] +  wp[1]* x + wp[2]*y + wp[3]*x*y
    Py = wp[4] +  wp[5]* x + wp[6]*y + wp[7]*x*y
    return (Px,Py)

# ===============
# calculate the pixel coordinates towards world xy (screen) coordinates
def foto_to_world(point_pho_pix):
    x = point_pho_pix[0]
    y = point_pho_pix[1]

    wp = world_photo_fit_params
    M = np.array([[wp[1], wp[2], wp[3]],[wp[5], wp[6], wp[7]]])
    Mi = np.linalg.pinv(M)
    f_min_o = np.matrix([x - wp[0], y - wp[4]])
    world = Mi * f_min_o.T

    return (world[0,0], world[1,0])

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

            else:
                # ADD REFERENCE
                if (len(ref_pho_pix) < 9):
                    ref_pho_pix.append((x,y))
                    sort_references()
                else :
                    # laser_centers.append((x,y))
                    print("too many reference marks already, first remove some")


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




# ///////////
# main script
# ///////////
loadJson()
image_show()
cv2.setMouseCallback(filename, mouse_call)

print("press q to QUIT")
print("      r to RESET VIEW")
print("      l to LASER MODEL")
print("      c to clear Laser MODEL")
print("      p to sort the references and fit photo-world correction model")
print("      s to sort and map the lasers to world coordinates")
print("      k to map laser to meters and show")
print("      m to perspective fit the reference photo-world correction model")
print("      CNTRL-LMB to REMOVE LASER and/or REFERENCE POINT (only scale 1")
print("      RMB to DRAG VIEW")
print("      LMB to SET MARKER")

run = True
while(run) :
    plot.pause(0.001)

    key = cv2.waitKey(30)
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
        laser_center_pix = []
        image_show()
    elif (key == ord("p")):
        sort_references()
        image_show()
    elif (key == ord("s")):
        laser_map()
        image_show()
    elif (key == ord("k")):
        laser_pho_pix_2_laser_wor_met()
        fig1 = plot.figure()
        x = [c[0] for c in laser_wor_met]
        y = [c[1] for c in laser_wor_met]
        plot.plot(x,y,'.')
        plot.ion()
        plot.show()
        image_show()

    elif (key == ord("m")):
        if (world_photo_perspective_model == 0):
            world_photo_perspective_model = PerspectiveModel()
        world_photo_perspective_model.fit(ref_wor_met, ref_pho_pix)

        print("world_photo_fit_params " + str(world_photo_fit_params))

        image_show()

        
saveJson()
cv2.destroyAllWindows()

