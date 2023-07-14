import cv2
import os
from utils import *

def main():
    parser = get_default_parser()
    parser.add_argument('-e', '--ext',
                        help='Filename extension of masks',
                        default='_cp_masks_filtered.png')
    parser.add_argument('-o', '--outfile',
                        help='Name of csv file to store cell diameter data',
                        default='cells.csv')
    args = parser.parse_args()
    with open(os.path.join(args.dir, args.outfile), 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['Name', 'Cell No.', 'Center X', 'Center Y', 'Width', 'Height', 'Angle'])
        for f in get_filenames(args):
            name = os.path.split(f)[-1]
            if args.verbose:
                print('Processing: ', f)
            outlines = get_outlines(open_masks(f), min_size=0, verbose=args.verbose)
            for i, outline in enumerate(outlines):
                (X, Y), (W, H), A = cv2.minAreaRect(outline)
                writer.writerow((name, i, X, Y, W, H, A))

if __name__=='__main__':
    main()
