import numpy as np
print("numpy version  : " + np.version.version)
import matplotlib.pyplot as plot
import cv2 as cv2
print("opencv version : " + cv2.__version__ )
import json
import argparse
import os.path

filename = "fotos/20210609_143107.JPG"
outer_pixel_setpoint = 360   # from the setting of the laser : the outer XY

json_file_name = filename + ".json"
img_org = cv2.imread(filename, cv2.IMREAD_COLOR)
img_height = img_org.shape[0]
img_width  = img_org.shape[1]
scale = 0.25
crop_select = False
crop          = [[0, img_width], [0, img_height]]   # [[x1,x2], [y1,y2]]
laser_centers = []
refer_marks   = []
DRAW_CIRCLE_RADIUS = 14
DRAW_REF_RADIUS    = 14

# =====================================
# load json file.
if (os.path.isfile(json_file_name)):
    with open(json_file_name, 'r', encoding='utf-8') as json_file:
        json_data = json_file.read()
    json_data = json.loads(json_data)
    refer_marks = json_data["refer_marks"]
    laser_centers = json_data["laser_centers"]


# =====================================
# save json file.
def saveJson():
    global refer_marks
    with open(json_file_name, 'w', encoding='utf-8') as json_file:
        json_data = {}
        json_data["refer_marks"] = refer_marks
        json_data["laser_centers"] = laser_centers
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)



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
    for r in refer_marks:
        counter += 1
        RR = int(DRAW_REF_RADIUS/2)
        cv2.drawMarker(img, (r[0],r[1]), color=(0,255,55),thickness=4 ) 
        cv2.putText(img, str(counter) ,(r[0]+4,r[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 5 , color=(0,255,55), thickness = 2)
        
    img = img[crop[1][0]:crop[1][1], crop[0][0]:crop[0][1]] #https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python#15589825
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    cv2.imshow(filename, img)

# =======================
# IMAGE PROCESS THE LASERS
def find_dots(image):
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
    if(len(refer_marks) != 9) :
        print("cannot sort reference, only when 9")
        return
    refer_marks.sort(key=x_value)
    refer_marks.sort(key=y_value)
    


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

    elif event == cv2.EVENT_LBUTTONDOWN:
        l_mouse_down = True
        print(" l mouse  " + str(x) + " " + str(y))
        x += crop[0][0]
        y += crop[1][0]
        if (scale == 1):
            if (flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON) :
                # REMOVE UNWANTED LASER OR REFERENCES
                print("CNTRL")
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
                    refer_marks.append((x+crop[0][0],y+crop[1][0]))
                    sort_references()
                else:
                    print("too many reference marks already, first remove some")


    elif (event == cv2.EVENT_LBUTTONUP):
        l_mouse_down = False
        print(" l button " + str(x))

    elif (event == cv2.EVENT_RBUTTONDOWN):
        r_mouse_down = True
        # crop_select = True
        r_mouse_down_last = [x,y]
        print(" r mouse  " + str(x) + " " + str(y))
    elif (event == cv2.EVENT_RBUTTONUP):
        r_mouse_down = False
        print(" r mouse  " + str(x) + " " + str(y))
        
    elif (event == cv2.EVENT_MOUSEWHEEL) :
        print(" m wheel  " + str(x) + " " + str(y) + " " + str(flags) + " " + str(param))
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
print("      p to sort the refernces")
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
        find_dots(img_org)
    elif (key == ord("p")):
        sort_references()
        image_show()
    elif (key == ord("t")):
        print( "REMOVE LASER POINT")
        
        
saveJson()
cv2.destroyAllWindows()

