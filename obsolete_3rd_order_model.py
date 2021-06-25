import cv2 as cv2
import numpy as np

# ==================================================================================
# model the perspective using end order distortion model (linear, least squares.)
# obsolete : doesn't work as good as needed, replaced by real persprective model
# ==================================================================================

world_photo_fit_params = []

def image_show(img, ref_wor_met, DRAW_CIRCLE_RADIUS):

    if (world_photo_fit_params ):
        for w in ref_wor_met:
            photo_coor = world_to_foto(w)
            cv2.circle(img, (int(photo_coor[0]), int(photo_coor[1])), int(DRAW_CIRCLE_RADIUS), color=(255, 0, 0), thickness=2) # (B, G, R)


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
def fit_world_photo(ref_pho_pix,ref_wor_met):
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