import cv2
import os
import numpy as np

from utils import *
from directionality import add_dir_args

def add_pairwise_args(parser):
    parser.add_argument('--max_dist', type=float,
                        help='Maximum distance to include',
                        default=1000.)
    parser.add_argument('--neighbors', type=int,
                        help='Number of neighbors',
                        default=2)
    parser.add_argument('--scaling_factor', type=int,
                        help='Scaling factor, higher values follow image directionality more',
                        default=10)
    parser.add_argument('--measure_from',
                        choices=['center', 'right', 'left', 'top', 'bottom'],
                        default='right',
                        help='Whether to start measuring at right edge of box')
    return parser

def main():
    parser = get_default_parser()
    parser = add_dir_args(parser)
    parser = add_pairwise_args(parser)
    parser.add_argument('--ext',
                        help='Filename extension of masks',
                        default='_cp_masks_filtered.png')
    parser.add_argument('--img_ext',
                        help='Filename extension of output visualization',
                        default='_vis.png')
    parser.add_argument("--visualize",
                        help="Include output visualization image",
                        action="store_false")
    parser.add_argument('--outfile',
                        help='Name of csv file to store cell pairs data',
                        default='pairs.csv')
    args = parser.parse_args()
    with open(get_full_path(args, args.outfile), 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['Name', 'Pair No.', 'X1', 'Y1', 'X2', 'Y2', 'Distance'])
        for f in get_filenames(args):
            if args.verbose:
                print('Processing: ', f)
            im = cv2.imread(f.replace(args.ext, '.jpg'))
            img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            if args.visualize:
                plt.figure(figsize=(10,6))
            angle = directionality(img, bins=args.bins,
                                   kde_bw_method=args.bw_method,
                                   ksize=args.kernel_size,
                                   verbose=args.verbose,
                                   plot=args.visualize)
            outlines = list(get_outlines(open_masks(f), min_size=0, verbose=args.verbose))
            draw_slope_lines(img, angle, nlines=args.nlines)
            pairs, img, *_ = find_pairs(outlines, im, angle,
                                        max_dist=args.max_dist, neighbors=args.neighbors,
                                        scaling_factor=args.scaling_factor,
                                        measure_from=args.measure_from)
            if args.visualize:
                plt.imshow(img)
                plt.tight_layout()
                plt.savefig(replace_ext(f, args.img_ext))
            for i, (c1, c2) in enumerate(pairs):
                (x1, y1), (x2, y2) = c1, c2
                writer.writerow((get_name(f), i, x1, y1, x2, y2, np.linalg.norm(c1-c2)))

if __name__=='__main__':
    main()
