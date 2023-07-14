#@ String(label='Directory to images') path
#@ String(label='Output filename', value='stitching_result.csv') outfile
#@ Integer(label='Number of peaks for stitching', value=50) peaks
#@ Boolean(label='Manually specify ROIs', value=False) manual


# Example:
#   To run on all images in "imgs" folder, output to "stitching_result.csv",
#   using 50 peaks (more peaks improve accuracy but increases processing time)
#
#   ./ImageJ-linux64 --ij2 --console --run stitching.py '[path="imgs",outfile="stitching_result.csv",peaks=50,manual=false]'

#   To run stitching manually specifying ROIs, run (10 peaks work better):
#   ./ImageJ-linux64 --ij2 --console --run stitching.py '[path="imgs",outfile="stitching_result.csv",peaks=10,manual=true]'


import glob
import os
import csv

from itertools import tee

from ij import IJ
from ij.plugin import HyperStackConverter
from mpicbg.stitching import PairWiseStitchingImgLib, StitchingParameters

from fiji.stacks import Hyperstack_rearranger

def pairwise(it):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(it)
    next(b, None)
    return zip(a, b)

def get_params(peaks, r_threshold):
    params = StitchingParameters()
    params.dimensionality = 2
    params.cpuMemChoice = 1 # More RAM, less time FIXME offer choice?
    params.channel1 = 0
    params.channel2 = 0
    params.timeSelect = 0
    params.checkPeaks = peaks
    params.regThreshold = r_threshold
    params.computeOverlap = True
    params.displayFusion = False
    params.subpixelAccuracy = True
    params.fusionMethod = 0 # Linear Blending
    params.outputDirectory = None
    return params

def load_img(img_path):
    #return HyperStackConverter.toHyperStack​(img, 3, 1, 1, 'xyczt','grayscale') # 3 channels, 1 slice, 1 frame
    return Hyperstack_rearranger.convertToHyperStack(IJ.openImage(img_path))

def image_filenames(path, wildcard='*.jpg'):
    return sorted(glob.glob(os.path.join(path, wildcard)))

def get_filename(path):
    return os.path.split(path)[-1]

def stitch(img1, img2, peaks=50):
    params = get_params(peaks=peaks, r_threshold=0.3)
    return PairWiseStitchingImgLib.stitchPairwise(
        img1, img2, img1.getRoi(), img2.getRoi(), 1, 1, params
    )

if __name__ in ['__builtin__','__main__']:
    with open(os.path.join(path, outfile), 'wb') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['Img 1', 'Img 2', 'X offset', 'Y offset', 'R value'])
        img_names = image_filenames(path)
        for img_name1, img_name2 in pairwise(img_names):
            print(img_name1, img_name2)
            im1 = load_img(img_name1)
            im2 = load_img(img_name2)
            if manual:
                IJ.runMacro('waitForUser("Draw ROI, then hit OK");')
            result = stitch(im1, im2, peaks)
            offset = result.getOffset()
            r_value = result.getCrossCorrelation()
            writer.writerow([get_filename(img_name1), get_filename(img_name2),
                             offset[0], offset[1], r_value])
            IJ.runMacro("close('*');")
