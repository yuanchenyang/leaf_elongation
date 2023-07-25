import math
from utils import *
from merge_imgs import add_merge_args, merge_imgs

def add_stitching_args(parser):
    parser.add_argument('--ext',
                        help='Filename extension of images',
                        default='.jpg')
    parser.add_argument('--no_merge',
                        help='Disable generating merged images',
                        action='store_true')
    parser.add_argument('--workers', type=int,
                        help='Number of CPU threads to use in FFT',
                        default=2)
    parser.add_argument('--min_overlap', type=int,
                        help='Set lower limit for overlapping region',
                        default=500000)
    parser.add_argument('--min_sample', type=int,
                        help='Regions with more than this amount of overlap will be subsampled for increased speed',
                        default=math.inf)
    parser.add_argument('--early_term_thresh', type=float,
                        help='Stop searching when correlation is above this value',
                        default=0.7)
    parser.add_argument('--use_wins', nargs="+", type=int,
                        help='Whether to try using Hanning window',
                        default=(0,))
    parser.add_argument('--peak_cutoff_std', type=float,
                        help='Number of standard deviations below max value to use for peak finding',
                        default=1)
    parser.add_argument('--peaks_dist_threshold', type=float,
                        help='Distance to consider as part of same cluster when finding peak centroid',
                        default=25)
    parser.add_argument('--filter_radius', nargs="+", type=int,
        default=(100,50,20), #(50,100,20,300,10,200,400,30,500,40),
        help='Low-pass filter radii to try, smaller matches coarser/out-of-focus features')
    return parser

def main():
    parser = add_stitching_args(add_merge_args(get_default_parser()))
    args = parser.parse_args()
    img_names = sorted(get_filenames(args))
    with open(get_full_path(args, args.stitching_result), 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['Img 1', 'Img 2', 'X offset', 'Y offset', 'Corr Value', 'Area', 'r', 'use_win'])
        for img_names in pairwise(img_names):
            if args.verbose: print('Stitching', *img_names)
            corr, res, (dx, dy), val, area, r, use_win = stitch(args, *map(read_img, img_names))
            img_name1, img_name2 = map(get_name, img_names)
            writer.writerow([img_name1, img_name2, dx, dy, corr, area, r, use_win])
            if not args.no_merge:
                res_dir = get_full_path(args, args.result_dir, mkdir=True)
                merge_imgs(args, res_dir, img_name1, img_name2, dx, dy)

if __name__=='__main__':
    main()
