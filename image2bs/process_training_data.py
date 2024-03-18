
from image2landmark.image2landmark import image2landmark
from landmark2exp.landmark2exp import landmark2exp

if __name__ == '__main__':
    print('\033[1;31m%ss\033[0m' % "# Phase 1: Images to landmark")
    image2landmark()
    print('\033[1;31m%ss\033[0m' % "# Phase 2: Landmarks to 3DMM coefficient")
    landmark2exp()


