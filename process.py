import numpy as np

import patches
import pipeline as p
import cv2
from math import inf

_debug = False
_ploty = None
_ool, _oor = None, None


def process_image(original):
    global _ploty, _ool, _oor, lleft_x, lleft_y, lright_x, lright_y

    dst = p.undistort(original)
    warped, sure = p.get_birdview(dst)
    edges = p.get_lines(warped)
    sm = np.array(edges[-edges.shape[0] // 4:, :])

    if _ploty is None:
        _ploty = np.linspace(0, sm.shape[0] - 1, num=sm.shape[0])

    left_x, left_y, right_x, right_y = patches.find_curves2(sm, _ool, _oor, de2=None, verbose=5 if _debug else 0)
    left_fitx, right_fitx, _ool, _oor = patches.polyfit(_ploty, left_x, left_y, right_x, right_y)

    if (_ool is not None) and (_oor is not None) and ((_oor - _ool > 16) or (_oor - _ool < 6)):
        _ool, _oor = None, None

    valid = False
    if left_fitx is not None and right_fitx is not None:
        left_curvature, right_curvature, calculated_deviation = p.calculate_info(left_x, left_y, right_x, right_y)
        if abs(calculated_deviation) < 0.6:
            valid = True
            lleft_x, lleft_y, lright_x, lright_y = left_x, left_y, right_x, right_y

    if not valid:
        left_x, left_y, right_x, right_y = lleft_x, lleft_y, lright_x, lright_y
        left_fitx, right_fitx, _ool, _oor = patches.polyfit(_ploty, left_x, left_y, right_x, right_y)
        left_curvature, right_curvature, calculated_deviation = p.calculate_info(left_x, left_y, right_x, right_y)

    layer = p.fill_lane_lines(np.dstack((edges, edges, edges)).astype('uint8'), np.add(_ploty, 300), left_fitx,
                              right_fitx)

    res = p.merge_marks(dst, layer)

    #
    if left_curvature > 99999:
        left_curvature = inf

    if right_curvature > 99999:
        right_curvature = inf

    curvature_text = 'Curvature left: {:.2f} m    right: {:.2f} m'.format(left_curvature, right_curvature)
    cv2.putText(res, curvature_text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (28, 221, 119), 2)

    deviation_info = 'Lane Deviation: {:.2f} m'.format(calculated_deviation)
    cv2.putText(res, deviation_info, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (28, 221, 119), 2)
    return res
