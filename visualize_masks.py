from PIL import Image
from utils import *
import cv2
import os
import numpy as np

def main():
    parser = get_default_parser()
    parser.add_argument('--ext',
                        help='Filename extension of masks',
                        default='_cp_masks.png')
    parser.add_argument('--out_ext',
                        help='Filename extension of output visualization',
                        default='_vis.png')
    parser.add_argument("--outline",
                        help="Draw outlines or filled images",
                        action="store_true")
    args = parser.parse_args()

    for f in get_filenames(args):
        imgfile = f.replace(args.ext, '.jpg')
        outfile = f.replace(args.ext, args.out_ext)
        if args.verbose:
            print('Processing: ', f)
        if args.outline:
            img = Image.open(imgfile)
            mask_on_img = draw_masks_on_img(img, masks_to_one_hot_array(open_masks(f)))
            to_pil_image(mask_on_img).save(outfile)
        else:
            img = cv2.imread(imgfile)
            plt.figure(figsize=(10,6))
            outlines = list(get_outlines(open_masks(f), min_size=0, verbose=args.verbose))
            cv2.drawContours(img, outlines, -1, (255, 0, 0), 3)
            plt.imshow(img)
            plt.tight_layout()
            plt.savefig(outfile)

if __name__=='__main__':
    main()
