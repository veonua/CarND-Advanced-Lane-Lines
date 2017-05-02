import matplotlib.image as mpimg
import process
from moviepy.editor import VideoFileClip


base = "/home/veon/edu/udacity/CarND-Advanced-Lane-Lines/test_images/"
# vid1 = [base + "vid1/021.jpg", base + "vid1/025.jpg", base + "vid1/026.jpg"]
# hard = [base + "vid2/hard009.jpg", base + "vid2/hard016.jpg", base + "vid2/hard017.jpg",
#         base + "vid2/hard018.jpg", base + "vid2/hard025.jpg", base + "vid2/hard029.jpg",
#         base + "vid2/hard030.jpg", base + "vid2/hard032.jpg", base + "vid2/hard040.jpg",
#         base + "vid2/hard046.jpg", base + "vid2/hard050.jpg",
#         base + "vid3/cachal047.jpg"]
# tests = [base + "straight_lines1.jpg", base + "test1.jpg", base + "test2.jpg", base + "test3.jpg", base + "test4.jpg",
#          base + "test5.jpg"]


image = False
process._debug = image

if image:
    o = mpimg.imread(base + "straight_lines1.jpg")
    process.process_image(o)
else:
    clip1 = VideoFileClip("./project_video.mp4")
    video_clip = clip1.fl_image(process.process_image)
    video_clip.write_videofile("./project_out.mp4", codec='mpeg4', audio=False)

