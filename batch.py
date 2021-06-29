from process import *
from traceback import format_exc

import pandas

result = pandas.DataFrame()

try:
    for laser_foto in laser_photos:
        print(f"Now processing {laser_foto.filename}")
        initialize_laser_las_pix(laser_foto)

        laser_map_and_sort(laser_foto)
        max_distances = model_projections(laser_foto, False)
        result = result.append({'file':laser_foto.filename, 'a':disto_transform_model.a, 'b':disto_transform_model.b, 'c':disto_transform_model.c,
                    'd':disto_transform_model.d, 'e':disto_transform_model.e, 's':disto_transform_model.s, 
                    't':disto_transform_model.t, 'x0':disto_transform_model.x0, "y0":disto_transform_model.y0,
                    'laser':laser_foto.laser_model, 'laser_axis':laser_foto.laser_axis, 'photo_axis':laser_foto.photo_axis, 
                    "max_residual[px]":max_distances}, ignore_index=True)
except Exception as e:
    track = format_exc()
    print(track)
finally:
    result.to_csv("disto_results.csv", index=False)


