import cv2
import os
import numpy as np
import networkx as nx

from utils import *
from directionality import add_dir_args
from cell_pairwise_dist import add_pairwise_args

def main():
    parser = get_default_parser()
    parser = add_dir_args(parser)
    parser = add_pairwise_args(parser)
    parser.add_argument('--ext',
                        help='Filename extension of masks',
                        default='_cp_masks_filtered.png')
    parser.add_argument('--ext_stomata',
                        help='Filename extension of masks',
                        default='_cp_masks_stomata.png')
    parser.add_argument('--new_ext',
                        help='Filename extension of masks filtered by stomata',
                        default='_by_stomata.png')
    parser.add_argument('--img_ext',
                        help='Filename extension of output visualization',
                        default='_vis.png')
    parser.add_argument("--no_visualize",
                        help="Include output visualization image",
                        action="store_true")
    parser.add_argument('--min_stomata', type=int,
                        help='Line must have at least this many stomata to be filtered',
                        default=3)
    args = parser.parse_args()
    for maskfile in get_filenames(args):
        if args.verbose:
            print('Processing: ', maskfile)
        im = cv2.imread(maskfile.replace(args.ext, '.jpg'))
        angle = directionality(im, bins=args.bins,
                               kde_bw_method=args.bw_method,
                               ksize=args.kernel_size,
                               verbose=args.verbose)
        stomata_maskfile = maskfile.replace(args.ext, args.ext_stomata)
        if not os.path.isfile(stomata_maskfile):
            if args.verbose:
                print(f"skipping {maskfile} as {stomata_maskfile} does not exist!")
            continue
        masks_cells, masks_stomata = map(open_masks, (maskfile, stomata_maskfile))
        outlines_cells, outlines_stomata = [list(get_outlines(m, min_size=0))
                                            for m in (masks_cells, masks_stomata)]


        outlines = outlines_cells + outlines_stomata
        pairs, _, _, pairs_indices = find_pairs(outlines, im.copy(), angle,
                                                max_dist=args.max_dist, neighbors=args.neighbors,
                                                scaling_factor=args.scaling_factor,
                                                measure_from=args.measure_from)

        graph = nx.Graph()
        cells, stomata = set(), set()
        for i in range(len(outlines_cells)):
            graph.add_node(i)
            cells.add(i)
        for j in range(len(outlines_cells), len(outlines)):
            graph.add_node(j)
            stomata.add(j)
        for i, j in pairs_indices:
            graph.add_edge(i, j)

        selected = set()
        for c in nx.connected_components(graph):
            n_stomata = sum(n in stomata for n in c)
            if n_stomata < args.min_stomata:
                selected |= c - stomata

        if not args.no_visualize:
            plt.figure(figsize=(10,6))
            img = im.copy()
            #draw_slope_lines(img, angle, nlines=args.nlines)
            for c1, c2 in pairs:
                cv2.line(img, c1, c2, (0, 0, 255), 3)
            cv2.drawContours(img, outlines_stomata, -1, (0, 255, 0), 3)
            cv2.drawContours(img, outlines_cells, -1, (255, 0, 0), 3)
            cv2.drawContours(img, [outlines_cells[i] for i in selected], -1, (255, 0, 0), 20)
            plt.imshow(img)
            plt.savefig(replace_ext(maskfile, args.img_ext))

        orig_indices = np.unique(masks_cells)[1:]
        orig_indices_selected = [orig_indices[i] for i in selected]
        new_masks = masks_cells.copy()
        new_masks[np.logical_not(np.isin(new_masks, orig_indices_selected))] = 0
        save_masks(replace_ext(maskfile, args.new_ext), new_masks)

if __name__=='__main__':
    main()
