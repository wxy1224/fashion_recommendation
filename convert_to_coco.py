from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.widgets import RadioButtons
from matplotlib.path import Path

from skimage.measure import find_contours

from PIL import Image
import matplotlib

import argparse
import numpy as np
import glob
import os

from matplotlib.widgets import Button
from matplotlib.lines import Line2D
from matplotlib.artist import Artist

from poly_editor import PolygonInteractor

from matplotlib.mlab import dist_point_to_segment
import sys
from visualize_dataset import return_info



if __name__=='__main__':
    image_dir = sys.argv[1]
    img_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

    # if len(self.img_paths) == 0:
    #     self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, '*.png')))
    # if os.path.exists(self.img_paths[self.index][:-3] + 'txt'):
    #     self.index = len(glob.glob(os.path.join(self.img_dir, '*.txt')))
