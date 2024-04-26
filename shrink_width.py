import cv2
import os
import networkx as nx

from utils import *


def main():
    parser = get_default_parser()
    parser.add_argument('--ext',
                        help='Filename extension of masks',
                        default='_cp_masks.png')
    parser.add_argument('--new_img_ext',
                        help='New extension of image files',
                        default='_shrunk.jpg')
    parser.add_argument('--new_mask_ext',
                        help='New extension of mask files',
                        default='_shrunk_cp_masks.png')
    parser.add_argument('--factor',
                        help='rescale factor',
                        default=4)
    args = parser.parse_args()
    for f in get_filenames(args):
        img = cv2.imread(f.replace(args.ext, '.jpg'))
        H, W, _ = img.shape
        img_new = cv2.resize(img, (W//args.factor, H),interpolation=cv2.INTER_AREA)
        cv2.imwrite(f.replace(args.ext, args.new_img_ext), img_new)

        mask = open_masks(f)
        mask_new = cv2.resize(mask, (W//args.factor, H), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f.replace(args.ext, args.new_mask_ext), mask_new)


if __name__=='__main__':
    main()
