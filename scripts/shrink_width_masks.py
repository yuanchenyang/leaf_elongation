import cv2
import os
from utils import *

def main():
    parser = get_default_parser()
    parser.add_argument('--ext',
                        help='Filename extension of masks',
                        default='_cp_masks.png')
    parser.add_argument('--new_mask_ext',
                        help='New extension of mask files',
                        default='_shrunk_cp_masks.png')
    parser.add_argument('--stretch',
                        help='use this flag to stretch instead of shink',
                        action='store_true')
    parser.add_argument('--factor',
                        type=int,
                        help='rescale factor',
                        default=4)
    args = parser.parse_args()
    for f in get_filenames(args):
        mask = open_masks(f)
        H, W = mask.shape
        new_W = W * args.factor if args.stretch else W // args.factor
        mask_new = cv2.resize(mask, (new_W, H), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f.replace(args.ext, args.new_mask_ext), mask_new)

if __name__=='__main__':
    main()
