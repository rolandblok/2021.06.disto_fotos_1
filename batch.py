from process import *
from traceback import format_exc

import pandas

result = pandas.DataFrame()

try:
    for file, start, step, mid, line, laser, camera in [("fotos_2/20210618_095352.JPG", 50, 50, 400, 15, 'on-axis', 'off-axis'),
                                        ("fotos_2/20210618_095410.JPG", 50, 50, 400, 15, 'on-axis', 'off-axis'),
                                        ("fotos_2/20210618_095516.JPG", 50, 50, 400, 15, '?', '?'),
                                        ("fotos_2/20210618_095534.JPG", 50, 50, 400, 15, '?', '?'),
                                        ("fotos_2/20210618_095609.JPG", 50, 50, 400, 15, '?', '?'),
                                        ("fotos_2/20210618_095618.JPG", 50, 50, 400, 15, '?', '?'),
                                        ("fotos_2/20210618_100631.JPG", 50, 50, 400, 15, '?', '?'),
                                        ("fotos_2/20210618_100642.JPG", 50, 50, 400, 15, '?', '?'),
                                        ("fotos_2/20210618_101129.JPG", 50, 100, 400, 8, '?', '?'),
                                        ("fotos_2/20210618_101220.JPG", 50, 100, 400, 8, '?', '?'),
                                        ("fotos_2/20210618_101230.JPG", 50, 100, 400, 8, '?', '?'),
                                        ("fotos_2/20210618_101335.JPG", 50, 100, 400, 8, '?', '?')
                                        ]:
        print(f"Now processing {file}")
        initialize_laser_las_pix(start, step, mid, line)
        json_file_name = file + ".json"
        loadJson(json_file_name)

        laser_map_and_sort()
        max_distances = model_projections(False)
        result = result.append({'file':file, 'a':disto_transform_model.a, 'b':disto_transform_model.b, 'c':disto_transform_model.c,
                    'd':disto_transform_model.d, 'e':disto_transform_model.e, 's':disto_transform_model.s, 
                    't':disto_transform_model.t, 'laser':laser, 'camera':camera, "max_residual[px]":max_distances}, ignore_index=True)
except Exception as e:
    track = format_exc()
    print(track)
finally:
    result.to_csv("disto_results.csv", index=False)


