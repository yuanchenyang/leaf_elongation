from utils import *

def main():
    parser = get_default_parser()
    parser.add_argument('-e', '--ext',
                        help='Filename extension of masks',
                        default='_cp_masks.png')
    parser.add_argument('-n', '--new_ext',
                        help='Filename extension of rois',
                        default='_rois.zip')
    parser.add_argument('-m', '--min_size', type=int,
                        help='Minimum mask area to include',
                        default=300)
    args = parser.parse_args()
    for f in get_filenames(args):
        if args.verbose:
            print('Processing: ', f)
        masks = open_masks(f)
        outlines = get_outlines(masks, args.min_size, verbose=args.verbose)
        save_rois(replace_ext(f, args.new_ext), outlines, verbose=args.verbose)

if __name__=='__main__':
    main()
