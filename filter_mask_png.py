from utils import *

def main():
    parser = get_default_parser()
    parser.add_argument('-e', '--ext',
                        help='Filename extension of masks',
                        default='_cp_masks.png')
    parser.add_argument('-n', '--new_ext',
                        help='Filename extension of filtered masks',
                        default='_filtered.png')
    parser.add_argument('-m', '--min_size', type=int,
                        help='Minimum mask area to include',
                        default=500)

    args = parser.parse_args()
    for f in get_filenames(args):
        if args.verbose:
            print('Processing: ', f)
        masks = open_masks(f)
        filtered = filter_masks(masks, min_size=args.min_size, verbose=args.verbose)
        save_masks(replace_ext(f, args.new_ext), filtered)

if __name__=='__main__':
    main()
