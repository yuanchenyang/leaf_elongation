import argparse
import os
import cv2
import glob
import csv
import numpy as np
import scipy.fft as sft

from functools import partial
from itertools import tee, product, chain
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors, NearestCentroid
from PIL import Image
from roifile import ImagejRoi, roiwrite
from matplotlib import pyplot as plt

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_filenames(args):
    return glob.glob(os.path.join(args.dir, f'*{args.ext}'))

def get_full_path(args, filename, mkdir=False):
    path = os.path.join(args.dir, filename)
    if mkdir and not os.path.exists(path):
        os.makedirs(path)
    return path

def get_name(path):
    return os.path.split(path)[-1]

def replace_ext(filename, new_ext):
    return os.path.splitext(filename)[0] + new_ext

def open_masks(filename):
    return np.array(Image.open(filename))

def save_masks(filename, masks):
    im = Image.fromarray(masks)
    im.save(filename)

def get_default_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir',
                        help='Base directory')
    parser.add_argument("-v", "--verbose",
                        help="Increase output verbosity",
                        action="store_true")
    return parser

def get_outlines(masks, min_size, verbose=False):
    """ get outlines of masks as a list to loop over for plotting """
    indices, counts = np.unique(masks, return_counts=True)
    if verbose:
        print('Total ROIs:', len(indices) - 1)
    for i, count in zip(indices, counts):
        if i == 0 or count < min_size:
            continue
        mn = masks==i
        contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        contours = contours[-2]
        cmax = np.argmax([c.shape[0] for c in contours])
        pix = contours[cmax].astype(int).squeeze()
        if len(pix) > 4:
            yield pix

def rot_matrix(t):
    return np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]])

def scale(x, y, angle, factor):
    x1, y1 = rot_matrix(-angle) @ np.array([x, y])
    x2, y2 = rot_matrix(angle) @ np.array([x1, y1 * factor])
    return x2, y2

def get_anchor(measure_from, rect):
    if measure_from == 'center':
        return rect[0]
    ps = cv2.boxPoints(rect)
    centers = np.array([(p1 + p2)/2 for p1, p2 in pairwise(np.vstack([ps, ps[0]]))])
    if measure_from == 'left':
        return centers[np.argmin(centers[:, 0])]
    if measure_from == 'right':
        return centers[np.argmax(centers[:, 0])]
    if measure_from == 'top':
        return centers[np.argmin(centers[:, 1])]
    if measure_from == 'bottom':
        return centers[np.argmax(centers[:, 1])]
    raise ValueError('Invalid `measure_from`!')

def find_pairs(outlines, img, outfile, angle,
               max_dist=1000, neighbors=2, scaling_factor=10, measure_from='center'):
    rects = [cv2.minAreaRect(o) for o in outlines]
    rect_anchors = [get_anchor(measure_from, r) for r in rects]
    scaled_anchors = [scale(x, y, angle, scaling_factor) for x, y in rect_anchors]
    if len(scaled_anchors) <= neighbors:
        return # Too few rects to find neighbors
    nbrs = NearestNeighbors(n_neighbors=neighbors+1, algorithm='ball_tree').fit(scaled_anchors)
    distances, indices = nbrs.kneighbors(scaled_anchors)
    cv2.drawContours(img, outlines, -1, (0, 255, 0), 3)
    cv2.drawContours(img, [np.intp(cv2.boxPoints(r)) for r in rects],
                     -1, (255, 0, 0), 3)
    for i, ((_, *idx), (_, *dist)) in enumerate(zip(indices, distances)):
        for j, d in zip(idx, dist):
            if d < max_dist: # Reduce to filter out far apart cells
                c1, c2 = np.intp(rect_anchors[i]), np.intp(rect_anchors[j])
                yield c1, c2
                cv2.line(img, c1, c2, (0, 0, 255), 3)
    plt.imshow(img)
    plt.tight_layout()
    plt.savefig(outfile)

def save_rois(filename, outlines, verbose=False):
    rois = [ImagejRoi.frompoints(outline) for outline in outlines]
    if verbose:
        print(f'Saving {len(rois)} ROIs')

    # Delete file if it exists; the roifile lib appends to existing zip files.
    if os.path.exists(filename):
        os.remove(filename)
    roiwrite(filename, rois)

def filter_masks(masks, min_size, verbose=False):
    indices, counts = np.unique(masks, return_counts=True)
    if verbose:
        print(f'{len(counts) - 1} total masks of sizes')
        print(np.sort(counts[1:]))
    for i, count in zip(indices, counts):
        if count < min_size:
            masks[masks == i] = 0
    if verbose:
        print(f'{len(np.unique(masks)) - 1} masks after filtering')
    return masks

def directionality(im, bins=500, kde_bw_method=0.10, ksize=7,
                   verbose=False, plot=False):
    grad_x = cv2.Sobel(src=im, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    grad_y = cv2.Sobel(src=im, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    magnitude = np.sqrt(grad_x**2 + grad_y**2).astype(np.float64)
    magnitude /= np.max(magnitude)
    angles = np.arctan2(grad_y, grad_x)
    # Range from -pi/4 to pi/4, so center at 0
    angles_shifted = ((angles + np.pi/4) % (np.pi/2) - np.pi/4)
    data = angles_shifted.flatten()
    # Remove 0 values from data
    data = data[data!=0]
    weights, p = np.histogram(data, bins=bins)
    kde = gaussian_kde(p[1:], weights=weights, bw_method=kde_bw_method)
    x = np.linspace(-np.pi/4, np.pi/4, bins*20)
    y = kde.pdf(x)
    angle = x[np.argmax(y)]
    if verbose: print(f'Found angle: {angle*180/np.pi} degrees')
    if plot:
        plt.hist(data, bins=bins, density=True)
        plt.plot(x, y)
    return angle

def draw_slope_lines(im_gray, angle, nlines=10):
    h, w = im_gray.shape
    x = np.linspace(0, w-1, 2)
    slope = np.tan(angle)
    for dy in np.linspace(0, h-slope*w-1, nlines):
        plt.plot(x, slope*x + dy, 'c:')

def round_int(s):
    return int(round(float(s),0))

def get_filter_mask(img, r):
    x, y = img.shape
    mask = np.zeros((x, y), dtype="uint8")
    cv2.circle(mask, (y//2, x//2), r, 255, -1)
    return mask

def read_img(filename):
    return np.float64(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

def apply_filter(img, filter_mask, inverse=False):
    res = sft.fftshift(img)
    if inverse:
        res[filter_mask != 0] = 0
    else:
        res[filter_mask == 0] = 0
    return sft.ifftshift(res)

def corr(a1, a2):
    if len(a1) == 0 or len(a2) == 0:
        return 0
    return np.corrcoef(a1, a2)[0,1]

def get_overlap(img1, img2, coords, min_overlap=0., min_sample=np.inf, sample_factor=10):
    dx, dy = coords
    assert img1.shape == img2.shape
    Y, X = img1.shape
    if dy >= 0 and dx >= 0:
        s1, s2 = img1[dy:Y, dx:X], img2[0:Y-dy, 0:X-dx]
    elif dy < 0 and dx >= 0:
        s1, s2 = img1[0:Y+dy, dx:X], img2[-dy:Y, 0:X-dx]
    else:
        return get_overlap(img2, img1, (-dx, -dy),
                           min_overlap=min_overlap,
                           min_sample=min_sample,
                           sample_factor=sample_factor)
    assert s1.shape == s2.shape
    area = s1.shape[0] * s1.shape[1]
    if area < min_overlap*Y*X:
        return -1, area
    f1, f2 = s1.flatten(), s2.flatten()
    if area > min_sample:
        print("sampled")
        indices = np.random.randint(0, area, size=area // sample_factor)
        f1, f2 = f1[indices], f2[indices]
    return corr(f1, f2), area

def round_int(x):
    return np.round(x).astype('int')

def centroids(coords, labels):
    for c in range(labels.max()+1):
        yield round_int(coords[labels == c].mean(axis=0))

def get_peak_centroids(args, res):
    #yield round_int(np.unravel_index(np.argmax(res), res.shape))
    #cutoff = res > (res.mean() + args.peak_cutoff_std * res.std())
    cutoff = res > (res.max() - args.peak_cutoff_std * res.std())
    if cutoff.sum() > 2:
        X = np.argwhere(cutoff)
        labels = AgglomerativeClustering(
            n_clusters=None,
            linkage='single',
            distance_threshold=args.peaks_dist_threshold
        ).fit(X).labels_
        cents = list(centroids(X, labels))
        yield from sorted(cents, key=lambda coord: res[tuple(coord)])
    else:
        yield from np.argwhere(cutoff)

def stitches(args, img1, img2):
    assert img1.shape == img2.shape
    win = cv2.createHanningWindow(img1.T.shape, cv2.CV_64F)
    Y, X = img1.shape
    for use_win in args.use_wins:
        f1, f2 = [sft.fft2(img * win if use_win else img,
                           norm='ortho', workers=args.workers)
                  for img in (img1, img2)]
        for r in args.filter_radius:
            mask = get_filter_mask(img1, r)
            G1, G2 = [apply_filter(f, mask) for f in (f1, f2)]
            R = G1 * np.ma.conjugate(G2)
            R /= np.absolute(R)
            res = sft.ifft2(R, norm='ortho', workers=args.workers).real
            for dy, dx in get_peak_centroids(args, res):
                for dX, dY in product((dx, -X+dx), (dy, -Y+dy)):
                    corr, area = get_overlap(img1, img2, (dX, dY),
                                             min_overlap=args.min_overlap,
                                             min_sample=args.min_sample)
                    if args.verbose:
                        print(f'dx:{dX: 5} dy:{dY: 5} corr:{corr:+f} area:{area: 9} r:{r: 3}')
                    yield corr, res, (dX, dY), res[dY, dX], area, r, use_win
                    if corr >= args.early_term_thresh:
                        return

def stitch(args, img1, img2):
    return max(stitches(args, img1, img2), key=lambda x: x[0]) # corr, res, idx, val
