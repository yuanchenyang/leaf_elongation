import cv2
import os
from segment_utils import *

def main():
    parser = get_default_parser()
    parser.add_argument('--ext',
                        help='Filename extension of images',
                        default='.jpg')
    img_names = sorted(get_filenames(args))
    args = parser.parse_args()



if __name__=='__main__':
    main()
