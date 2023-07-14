import cv2
import os
from utils import *
from matplotlib import pyplot as plt

def add_dir_args(parser):
    parser.add_argument('--draw_lines',
                        help='Draw lines on image and save as output',
                        action='store_false')
    parser.add_argument('--nlines', type=int,
                        help='Number of lines to draw on image',
                        default=10)
    parser.add_argument('--bins', type=int,
                        help='Number of bins to use in angles histogram',
                        default=500)
    parser.add_argument('--kernel_size', type=int,
                        help='Size of Sobel kernel to find direction',
                        default=7)
    parser.add_argument('--bw_method', type=float,
                        help='Gaussan kde parameter, controls the smoothness of curve-fit',
                        default=0.1)
    return parser

def main():
    parser = add_dir_args(get_default_parser())
    parser.add_argument('-e', '--ext',
                        help='Filename extension of images',
                        default='.jpg')
    parser.add_argument('-o', '--outfile',
                        help='Name of csv file to store cell directionality data',
                        default='dirs.csv')
    parser.add_argument('-p', '--plot',
                        help='Generate plots of direction histogram',
                        action='store_false')

    args = parser.parse_args()
    with open(os.path.join(args.dir, args.outfile), 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['Name', 'Angle (radians)'])
        for f in get_filenames(args):
            name = os.path.split(f)[-1]
            base_name = os.path.splitext(f)[0]
            if args.verbose:
                print('Processing: ', f)
            im = cv2.imread(f)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            plt.clf()
            angle = directionality(im, bins=args.bins,
                                   kde_bw_method=args.bw_method,
                                   ksize=args.kernel_size,
                                   verbose=args.verbose,
                                   plot=args.plot)
            writer.writerow((name, angle))
            if args.plot:
                plt.savefig(base_name + '_plot.png')
            if args.draw_lines:
                plt.clf()
                plt.figure(figsize=(10,6))
                plt.imshow(im, cmap='gray', vmin=0, vmax=255)
                draw_slope_lines(im, angle, nlines=args.nlines)
                plt.tight_layout()
                plt.savefig(base_name + '_lines.png')

if __name__=='__main__':
    main()
