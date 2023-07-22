from utils import *
from merge_imgs import add_merge_args, merge_imgs

def main():
    parser = add_merge_args(get_default_parser())
    parser.add_argument('--ext',
                        help='Filename extension of images',
                        default='.jpg')
    parser.add_argument('--no_merge',
                        help='Disable generating merged images',
                        action='store_true')
    parser.add_argument('--num_workers', type=int,
                        help='Number of CPU threads to use in FFT',
                        default=2)
    parser.add_argument('--min_overlap', type=int,
                        help='Set lower limit for overlapping region',
                        default=100000)
    parser.add_argument('--early_term_thresh', type=float,
                        help='Set lower limit for overlapping region',
                        default=0.5)
    parser.add_argument('--use_wins', nargs="+", type=int,
                        help='Whether to try using Hanning window',
                        default=(0,1))
    parser.add_argument('--filter_radius', nargs="+", type=int,
        default=(50,100,20,300,10,200,400,30,500,40),
        help='Low-pass filter radii to try, smaller matches coarser/out-of-focus features')
    args = parser.parse_args()

    img_names = sorted(get_filenames(args))
    with open(get_full_path(args, args.stitching_result), 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')

        writer.writerow(['Img 1', 'Img 2', 'X offset', 'Y offset', 'Corr Value'])
        for img_names in pairwise(img_names):
            if args.verbose: print('Stitching', *img_names)
            corr, res, (dx, dy), val = stitch(*map(read_img, img_names),
                                              rs=args.filter_radius,
                                              workers=args.num_workers,
                                              min_overlap=args.min_overlap)
            if args.verbose: print(dx, dy, corr, val)
            img_name1, img_name2 = map(get_name, img_names)
            writer.writerow([img_name1, img_name2, dx, dy, corr])
            if not args.no_merge:
                res_dir = get_full_path(args, args.result_dir, mkdir=True)
                merge_imgs(args, res_dir, img_name1, img_name2, dx, dy)

if __name__=='__main__':
    main()
