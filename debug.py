import matplotlib.image as mpimg
import numpy as np

import pipeline as p
import patches
from moviepy.editor import VideoFileClip

ploty = None
ool, oor = None, None

image = True

def process_image(original):
    global ploty, ool, oor

    dst = p.undistort(original)
    warped, sure = p.get_birdView(dst)  # np.divide(.astype(float), 255.0)
    edges = p.get_lines(warped)
    sm = np.array(edges[-edges.shape[0] // 4:, :])

    if ploty is None:
        ploty = np.linspace(0, sm.shape[0] - 1, num=sm.shape[0])

    left_fitx, right_fitx, ff, ool, oor = patches.find_curves(sm, ploty, de2=None, verbose= 5 if image else 0)

    if (ool is not None) and (oor is not None) and ((oor - ool > 16) or (oor - ool < 5)):
        ool, oor = None, None

    if left_fitx is None or right_fitx is None:
        return dst

    layer = p.fill_lane_lines(np.dstack((edges, edges, edges)).astype('uint8'), np.add(ploty, 300), left_fitx, right_fitx)

    return p.merge_marks(dst, layer)

if image:
    o = mpimg.imread("/home/veon/edu/udacity/CarND-Advanced-Lane-Lines/test_images/vid1/226.jpg")
    process_image(o)
else:
    clip1 = VideoFileClip("./harder_challenge_video.mp4")
    video_clip = clip1.fl_image(process_image)
    video_clip.write_videofile("./harder_challenge_out.mp4", codec='mpeg4', audio=False)

zzz = 1
