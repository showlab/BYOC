import os

from exp2bs.inference_exp2bs import inference_exp2bs
from image2landmark.image2landmark import image2landmark
from landmark2exp.landmark2exp import landmark2exp
from read_pose_angle import read_pose_angle
import time

from video2images import video2images

if __name__ == '__main__':
    if os.listdir("./test_video"):
        print('\033[1;31m%ss\033[0m' % "# Phase 0: Video to image")
        video2images()

    start = time.time()
    print('\033[1;31m%ss\033[0m' % "# Phase 1: Images to landmark")
    image2landmark()
    print('\033[1;31m%ss\033[0m' % "# Phase 2: Landmarks to 3DMM coefficient")
    landmark2exp()
    print('\033[1;31m%ss\033[0m' % "# Phase 3: 3DMM coefficients to blendshape")
    inference_exp2bs()
    print('\033[1;31m%ss\033[0m' % "# Phase 4: Reading pose angle")
    read_pose_angle()
    end = time.time()
    print("The cost of time is %ss." % str(end - start))


