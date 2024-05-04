import cv2
import os
from utils import *


def main():
    parser = get_default_parser()
    parser.add_argument('--ext',
                        help='Filename extension of image files',
                        default='.jpg')
    parser.add_argument('--new_img_ext',
                        help='New extension of image files',
                        default='_shrunk.jpg')
    parser.add_argument('--stretch',
                        help='use this flag to stretch instead of shink',
                        action='store_true')
    parser.add_argument('--factor',
                        type=int,
                        help='rescale factor',
                        default=4)
    args = parser.parse_args()
    for f in get_filenames(args):
        img = cv2.imread(f)
        H, W, _ = img.shape
        new_W = W * args.factor if args.stretch else W // args.factor
        img_new = cv2.resize(img, (new_W, H),interpolation=cv2.INTER_AREA)
        cv2.imwrite(f.replace(args.ext, args.new_img_ext), img_new)

if __name__=='__main__':
    main()
