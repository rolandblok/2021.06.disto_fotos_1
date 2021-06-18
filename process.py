import numpy as np
print("numpy version  : " + np.version.version)
import matplotlib.pyplot as plot
from json import JSONEncoder
import cv2 as cv2
print("opencv version : " + cv2.__version__ )
import json
import argparse
import os.path

# filename = "fotos/20210609_143107.JPG"
# outer_pixel_setpoint     = 360   # from the setting of the laser : the outer XY
# no_laser_splots_per_line = 37
filename = "fotos_2/20210618_095352.JPG"
outer_pixel_setpoint       = 400 - 50
no_laser_splots_per_line   = 15


world_coor = [[-1, 1], [0, 1], [1,1], # wereld coordinaten in meters.
              [-1, 0], [0, 0], [1,0], 
              [-1, -1],[0, -1],[1,-1]]

corner_i_map = {"tl" : 0,
                "tm" : 1,
                "tr" : 2,
                "ml" : 3,
                "mm" : 4,
                "mr" : 5,
                "bl" : 6,
                "bm" : 7,
                "br" : 8 }
laser_corners = []

json_file_name = filename + ".json"
img_org = cv2.imread(filename, cv2.IMREAD_COLOR)
img_height = img_org.shape[0]
img_width  = img_org.shape[1]
scale = 0.25
crop_select = False
crop          = [[0, img_width], [0, img_height]]   # [[x1,x2], [y1,y2]]
laser_centers = []
laser_centers_pix = []
laser_centers_wor = []
refer_marks   = []
world_photo_fit_params = []
DRAW_CIRCLE_RADIUS = 14
DRAW_REF_RADIUS    = 14

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# ===================================================
# what do/should we do : map two coordinate systems
#  laser_pixel ===> world_meters ===> photo_pixels
#
#  1st : manual create reference world coordinates (world --> photo pixels)
#    fit the world meters to photo pixels 
#  2nd : find the laser spots via image processing
#    map those to laser pixel grid.
#  3rd : convert laser-photo pixels --> world coordinates
#
#  4th : fit laser pixel grid towards world coordinates.




# =====================================
# load json file.
if (os.path.isfile(json_file_name)):
    with open(json_file_name, 'r', encoding='utf-8') as json_file:
        json_data = json_file.read()
    json_data = json.loads(json_data)
    refer_marks = json_data["refer_marks"]
    laser_centers = json_data["laser_centers"]
    laser_corners = json_data["laser_corners"]


# =====================================
# save json file.
def saveJson():
    global refer_marks
    with open(json_file_name, 'w', encoding='utf-8') as json_file:
        json_data = {}
        json_data["refer_marks"] = refer_marks
        json_data["laser_centers"] = laser_centers
        json_data["laser_corners"] = laser_corners
        json.dump(json_data, json_file, ensure_ascii=False, indent=4, cls=NumpyArrayEncoder)



# =====================================
# display the image.
def image_show():
    img = img_org.copy()
    counter = 0
    for center in laser_centers:
        cv2.circle(img, (int(center[0]), int(center[1])), int(DRAW_CIRCLE_RADIUS), color=(0, 0, 255), thickness=2) # (B, G, R)
        cv2.putText(img, str(counter) ,(center[0]+4,center[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 , color=(0,0,55), thickness = 1)
        counter += 1
    
    counter = 0
    for center in laser_centers_pix:
        cv2.circle(img, (int(center[0]), int(center[1])), int(DRAW_CIRCLE_RADIUS+5), color=(0, 255, 255), thickness=2) # (B, G, R)
        cv2.putText(img, str(counter) ,(center[0]+4,center[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 , color=(0,0,55), thickness = 1)
        counter += 1

    counter = 0
    for c in laser_corners:
        cv2.rectangle(img, (c[0]-DRAW_CIRCLE_RADIUS,c[1]-DRAW_CIRCLE_RADIUS),(c[0]+DRAW_CIRCLE_RADIUS,c[1]+DRAW_CIRCLE_RADIUS), color=(255,255,0), thickness = 2)
        cv2.putText(img, str(counter) ,(c[0]+4,c[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 , color=(0,0,55), thickness = 1)
        counter += 1

    counter = 0
    for r in refer_marks:
        counter += 1
        RR = int(DRAW_REF_RADIUS/2)
        cv2.drawMarker(img, (r[0],r[1]), color=(0,255,55),thickness=1 ) 
        cv2.putText(img, str(counter) ,(r[0]+4,r[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 5 , color=(0,255,55), thickness = 2)

    if (world_photo_fit_params):
        for w in world_coor:
            photo_coor = world_to_foto(w[0], w[1])
            cv2.circle(img, (int(photo_coor[0]), int(photo_coor[1])), int(DRAW_CIRCLE_RADIUS), color=(255, 0, 0), thickness=2) # (B, G, R)

        
    img = img[crop[1][0]:crop[1][1], crop[0][0]:crop[0][1]] #https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python#15589825
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    cv2.imshow(filename, img)

# =======================
# IMAGE PROCESS THE LASERS
def image_process_laser_dots(image):
    # https://stackoverflow.com/questions/51846933/finding-bright-spots-in-a-image-using-opencv#51848512
        #  constants
    BINARY_THRESHOLD = 100
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
    global laser_centers
    laser_centers = []
    components3 = components[3]
    for component in components3:
        # transform to serializable data
        laser_centers.append((int(component[0]), int(component[1])))
    image_show()

# =============
# sort the references (asume it's nine of them (3X3))
def x_value(v) : return v[0]
def y_value(v) : return v[1]
def sort_references():
    global refer_marks
    if(len(refer_marks) != 9) :
        print("cannot sort reference, only when 9")
        return
    refer_marks = sorted(refer_marks, key=y_value)
    for i in [0,1,2]:
        print(str(i))
        # refer_marks[3*i:(3*i)+3].sort(key=x_value)
        refer_marks[3*i:(3*i)+3] = sorted(refer_marks[3*i:(3*i)+3], key=x_value)
    

# =============
# helper function : closest point : return index
def closest_node(point, points):
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    return np.argmin(dist_2)



# =============
# map and sort the lasers
def det_laser_corners():
    # find the corners
    x_plus_y = [c[0]+c[1] for c in laser_centers]
    x_min_y  = [c[0]-c[1] for c in laser_centers]
    
    min_value = min(x_plus_y)
    min_index_x_plus_y = x_plus_y.index(min_value)
    min_value = min(x_min_y)
    min_index_x_min_y = x_min_y.index(min_value)

    max_value = max(x_plus_y)
    max_index_x_plus_y = x_plus_y.index(max_value)
    max_value = max(x_min_y)
    max_index_x_min_y = x_min_y.index(max_value)

    global laser_corners
    laser_corners = np.zeros((9,2))
    laser_corners[corner_i_map["tl"]] =  laser_centers[min_index_x_plus_y]
    laser_corners[corner_i_map["tr"]] =  laser_centers[max_index_x_min_y]
    laser_corners[corner_i_map["bl"]] =  laser_centers[min_index_x_min_y]
    laser_corners[corner_i_map["br"]] =  laser_centers[max_index_x_plus_y]

    print (" laser_corners " + str(laser_corners))
    
    laser_corners

# ====================
# Sort the laser corners
def sort_laser_corners():
    
    if(len(laser_corners) != 9) :
        print("cannot sort reference, only when 9")
        return
    laser_corners.sort(key=y_value)
    for i in [0,1,2]:
        laser_corners[3*i:3*i+3].sort(key=x_value)

    return

# =============
# map and sort the lasers
def laser_map():
    # create the result arrays
    global laser_centers_pix, laser_centers_wor
    laser_centers_pix = []
    laser_centers_wor = []
    counter = 0

    # create world coordinates line A, incl corners (exclude the last one.)
    Awxs = np.linspace(-outer_pixel_setpoint, outer_pixel_setpoint, no_laser_splots_per_line).round().astype(int)
    Awys = np.zeros(no_laser_splots_per_line)
    Aws  = np.dstack((Awxs[:-1], Awys[:-1]))[0] # form 2d point array, exclude last corner (it will be part of next line);  [0]: remove 3rd dimension
    # create laser estimates line A, incl corners
    Les = np.linspace(laser_corners[corner_i_map["tl"]], laser_corners[corner_i_map["tr"]], no_laser_splots_per_line).round().astype(int)
    # find the closest on line A 
    for Le in Les:
        close_le_i = closest_node(Le, laser_centers)
        laser_centers_pix.append(laser_centers[close_le_i])
    laser_centers_pix = Les
    laser_centers_wor.append(Aws)

    # debug




    return 

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
def fit_world_photo():
    global world_photo_fit_params

    M = np.zeros((18,8))
    P = np.zeros((18,1))
    row_cnt = 0
    for p,w in zip(refer_marks,world_coor):
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

    Ox, Oy, xMx, xMy, xMxy, yMx, yMy, yMxy  = np.linalg.lstsq(M, P, rcond=None)[0]
    world_photo_fit_params=(*Ox, *Oy, *xMx, *xMy, *xMxy, *yMx, *yMy, *yMxy)

    return

# ===============
# calculate the world coordinates towards photo pixel coordinates
def world_to_foto(x,y):
    wp = world_photo_fit_params
    Px = wp[0] +  wp[1]* x + wp[2]*y + wp[3]*x*y
    Py = wp[4] +  wp[5]* x + wp[6]*y + wp[7]*x*y
    return (Px,Py)

# ===============
# calculate the pixel coordinates towards world xy (screen) coordinates
def foto_to_world(x,y):
    wp = world_photo_fit_params
    M = np.array([[wp[0], wp[1]],[wp[2], wp[3]]])
    Mi = np.linalg.inv(M)
    f_min_o = np.matrix([x - wp[4], y - wp[5]])
    world = Mi * f_min_o.T

    return (world[0,0], world[1,0])

# ===============
# MOUSE HANDLING
l_mouse_down = False
r_mouse_down = False
r_mouse_down_last = [0,0]
def mouse_call(event, x, y, flags, param):
    global l_mouse_down
    global r_mouse_down
    global scale, img_height, img_width, crop, r_mouse_down_last
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
                for center in laser_corners:
                    # print(str(center) + str((x,y)))
                    if ((abs(x - int(center[0])) < DRAW_CIRCLE_RADIUS) and 
                        (abs(y - int(center[1])) < DRAW_CIRCLE_RADIUS)):
                        print ("HIT laser_corners " + str(cntr_cnt))
                        laser_corners.pop(cntr_cnt)
                        break
                    cntr_cnt += 1


                cntr_cnt = 0
                for center in laser_centers:
                    # print(str(center) + str((x,y)))
                    if ((abs(x - int(center[0])) < DRAW_CIRCLE_RADIUS) and 
                        (abs(y - int(center[1])) < DRAW_CIRCLE_RADIUS)):
                        print ("HIT laser " + str(cntr_cnt))
                        laser_centers.pop(cntr_cnt)
                        break
                    cntr_cnt += 1

                cntr_cnt = 0
                for center in refer_marks:
                    # print(str(center) + str((x,y)))
                    if ((abs(x - int(center[0])) < DRAW_CIRCLE_RADIUS) and 
                        (abs(y - int(center[1])) < DRAW_CIRCLE_RADIUS)):
                        print ("HIT reference marker")
                        refer_marks.pop(cntr_cnt)
                        break
                    cntr_cnt += 1

            else:
                # ADD REFERENCE
                if (len(refer_marks) < 9):
                    refer_marks.append((x,y))
                    sort_references()
                elif (len(laser_corners) < 9):
                    laser_corners.append((x,y))
                    sort_laser_corners()
                    print(str(laser_corners))
                else:
                    print("too many reference marks or laser_corners already, first remove some")


    elif (event == cv2.EVENT_LBUTTONUP):
        l_mouse_down = False
        # print(" l button " + str(x))

    elif (event == cv2.EVENT_RBUTTONDOWN):
        r_mouse_down = True
        # crop_select = True
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
image_show()
cv2.setMouseCallback(filename, mouse_call)

print("press q to QUIT")
print("      r to RESET VIEW")
print("      l to LASER MODEL")
print("      p to sort the references")
print("      s to map the lasers to world coordinates")
print("      f to fit reference : pixels to the world")
print("      y to test couple of laser centers (87 and 11)")
print("      CNTRL-LMB to REMOVE LASER and/or REFERENCE POINT (only scale 1")
print("      RMB to DRAG VIEW")
print("      LMB to SET MARKER")

run = True
while(run) :
    key = cv2.waitKey(0)
    print(str(key))
    if (key == ord("q")) : 
        run = False
    elif (key == ord("r")):
        crop          = [[0, img_width], [0, img_height]]   # img[y:y+h, x:x+w]
        scale = 0.25
        image_show()
    elif (key == ord("l")):
        print( "MODELING LASERS")
        image_process_laser_dots(img_org)
    elif (key == ord("p")):
        sort_references()
        image_show()
    elif (key == ord("s")):
        det_laser_corners()
        laser_map()
        image_show()
    elif (key == ord("f")):
        fit_world_photo()
        image_show()
    elif (key == ord("y")):
        print(str(foto_to_world(*laser_centers[87])))
        print(str(foto_to_world(*laser_centers[11])))


    elif (key == ord("t")):
        print( "REMOVE LASER POINT")
        
        
saveJson()
cv2.destroyAllWindows()

