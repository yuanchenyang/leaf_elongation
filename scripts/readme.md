## Stitching images

Stitch to get the relative locations of sequential two images in the folder

```console
$ python stitching.py --help
usage: stitching.py [-h] [-v] [-s STITCHING_RESULT] [-d RESULT_DIR] [-r] [--ext EXT] [--no_merge]
                    [--workers WORKERS] [--min_overlap MIN_OVERLAP] [--min_sample MIN_SAMPLE]
                    [--early_term_thresh EARLY_TERM_THRESH] [--use_wins USE_WINS [USE_WINS ...]]
                    [--peak_cutoff_std PEAK_CUTOFF_STD] [--peaks_dist_threshold PEAKS_DIST_THRESHOLD]
                    [--filter_radius FILTER_RADIUS [FILTER_RADIUS ...]]
                    dir

positional arguments:
  dir                   Base directory

options:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity (default: False)
  -s STITCHING_RESULT, --stitching_result STITCHING_RESULT
                        Stitching result csv file (default: stitching_result.csv)
  -d RESULT_DIR, --result_dir RESULT_DIR
                        Directory to save merged files (default: merged)
  -r, --exclude_reverse
                        Whether to additionally include img2 on top of img1 (default: False)
  --ext EXT             Filename extension of images (default: .jpg)
  --no_merge            Disable generating merged images (default: False)
  --workers WORKERS     Number of CPU threads to use in FFT (default: 2)
  --min_overlap MIN_OVERLAP
                        Set lower limit for overlapping region as a fraction of total image area (default: 0.125)
  --min_sample MIN_SAMPLE
                        Regions with more than this amount of overlap will be subsampled for increased speed
                        (default: inf)
  --early_term_thresh EARLY_TERM_THRESH
                        Stop searching when correlation is above this value (default: 0.7)
  --use_wins USE_WINS [USE_WINS ...]
                        Whether to try using Hanning window (default: (0,))
  --peak_cutoff_std PEAK_CUTOFF_STD
                        Number of standard deviations below max value to use for peak finding (default: 1)
  --peaks_dist_threshold PEAKS_DIST_THRESHOLD
                        Distance to consider as part of same cluster when finding peak centroid (default: 25)
  --filter_radius FILTER_RADIUS [FILTER_RADIUS ...]
                        Low-pass filter radii to try, smaller matches coarser/out-of-focus features (default:
                        (100, 50, 20))
```


## Merging images

Run after stitching to see stitching results

```console
$ python merge_imgs.py --help
usage: merge_imgs.py [-h] [-v] [-s STITCHING_RESULT] [-d RESULT_DIR] [-r] dir

positional arguments:
  dir                   Base directory

options:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity (default: False)
  -s STITCHING_RESULT, --stitching_result STITCHING_RESULT
                        Stitching result csv file (default: stitching_result.csv)
  -d RESULT_DIR, --result_dir RESULT_DIR
                        Directory to save merged files (default: merged)
  -r, --exclude_reverse
                        Whether to additionally include img2 on top of img1 (default: False)
```


## Image directionality

Plot distribution of edges in leaf and infer leaf directionality from this distribution.

```console
$ python directionality.py --help
usage: directionality.py [-h] [-v] [--draw_lines] [--nlines NLINES] [--bins BINS] [--kernel_size KERNEL_SIZE]
                         [--bw_method BW_METHOD] [-e EXT] [-o OUTFILE] [-p]
                         dir

positional arguments:
  dir                   Base directory

options:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity (default: False)
  --draw_lines          Draw lines on image and save as output (default: True)
  --nlines NLINES       Number of lines to draw on image (default: 10)
  --bins BINS           Number of bins to use in angles histogram (default: 500)
  --kernel_size KERNEL_SIZE
                        Size of Sobel kernel to find direction (default: 7)
  --bw_method BW_METHOD
                        Gaussan kde parameter, controls the smoothness of curve-fit (default: 0.1)
  -e EXT, --ext EXT     Filename extension of images (default: .jpg)
  -o OUTFILE, --outfile OUTFILE
                        Name of csv file to store cell directionality data (default: dirs.csv)
  -p, --plot            Generate plots of direction histogram (default: True)
```


## Cell diameters

Get cell sizes, extract ROI coordinates and ferret’s diameters

```console
$ python cell_diameter.py --help
usage: cell_diameter.py [-h] [-v] [-e EXT] [-o OUTFILE] dir

positional arguments:
  dir                   Base directory

options:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity (default: False)
  -e EXT, --ext EXT     Filename extension of masks (default: _cp_masks_filtered.png)
  -o OUTFILE, --outfile OUTFILE
                        Name of csv file to store cell diameter data (default: cells.csv)
```


## Pairwise distances

Measure two trichome(or any ROI) distances, can change scaling factors to make sure along leaf axis, can also check results and adjust parameters

```console
$ python cell_pairwise_dist.py --help
usage: cell_pairwise_dist.py [-h] [-v] [--draw_lines] [--nlines NLINES] [--bins BINS] [--kernel_size KERNEL_SIZE]
                             [--bw_method BW_METHOD] [--max_dist MAX_DIST] [--neighbors NEIGHBORS]
                             [--scaling_factor SCALING_FACTOR] [--measure_from {center,right,left,top,bottom}]
                             [--ext EXT] [--img_ext IMG_EXT] [--visualize] [--outfile OUTFILE]
                             dir

positional arguments:
  dir                   Base directory

options:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity (default: False)
  --draw_lines          Draw lines on image and save as output (default: True)
  --nlines NLINES       Number of lines to draw on image (default: 10)
  --bins BINS           Number of bins to use in angles histogram (default: 500)
  --kernel_size KERNEL_SIZE
                        Size of Sobel kernel to find direction (default: 7)
  --bw_method BW_METHOD
                        Gaussan kde parameter, controls the smoothness of curve-fit (default: 0.1)
  --max_dist MAX_DIST   Maximum distance to include (default: 1000.0)
  --neighbors NEIGHBORS
                        Number of neighbors (default: 2)
  --scaling_factor SCALING_FACTOR
                        Scaling factor, higher values follow image directionality more (default: 10.0)
  --measure_from {center,right,left,top,bottom}
                        Whether to start measuring at right edge of box (default: right)
  --ext EXT             Filename extension of masks (default: _cp_masks_filtered.png)
  --img_ext IMG_EXT     Filename extension of output visualization (default: _vis.png)
  --visualize           Include output visualization image (default: True)
  --outfile OUTFILE     Name of csv file to store cell pairs data (default: pairs.csv)
```


## Filter masks

Remove small masks (usually noise) of mask-like files in the folder, check help to change pattern of files

```console
$ python filter_mask_png.py --help
usage: filter_mask_png.py [-h] [-v] [-e EXT] [-n NEW_EXT] [-m MIN_SIZE] [-M MAX_SIZE] dir

positional arguments:
  dir                   Base directory

options:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity (default: False)
  -e EXT, --ext EXT     Filename extension of masks (default: _cp_masks.png)
  -n NEW_EXT, --new_ext NEW_EXT
                        Filename extension of filtered masks (default: _filtered.png)
  -m MIN_SIZE, --min_size MIN_SIZE
                        Minimum mask area to include (default: 500)
  -M MAX_SIZE, --max_size MAX_SIZE
                        Maximum mask area to include (default: 1000000)
```


## Shrink images

Shrink images by reducing their width. This is usually needed if the cells are
too long and have too high an aspect ratio where they cannot be recognized by Cellpose.

```console
$ python shrink_width.py --help
usage: shrink_width.py [-h] [-v] [--ext EXT] [--new_img_ext NEW_IMG_EXT] [--stretch] [--factor FACTOR] dir

positional arguments:
  dir                   Base directory

options:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity (default: False)
  --ext EXT             Filename extension of image files (default: .jpg)
  --new_img_ext NEW_IMG_EXT
                        New extension of image files (default: _shrunk.jpg)
  --stretch             use this flag to stretch instead of shink (default: False)
  --factor FACTOR       rescale factor (default: 4)
```


## Visualize masks

```console
$ python visualize_masks.py --help
usage: visualize_masks.py [-h] [-v] [--ext EXT] [--out_ext OUT_EXT] [--outline] dir

positional arguments:
  dir                Base directory

options:
  -h, --help         show this help message and exit
  -v, --verbose      Increase output verbosity (default: False)
  --ext EXT          Filename extension of masks (default: _cp_masks.png)
  --out_ext OUT_EXT  Filename extension of output visualization (default: _vis.png)
  --outline          Draw outlines or filled images (default: False)
```


## Filter sister cells using stomata

```console
$ python filter_sister_cells_using_stomata.py --help
usage: filter_sister_cells_using_stomata.py [-h] [-v] [--draw_lines] [--nlines NLINES] [--bins BINS]
                                            [--kernel_size KERNEL_SIZE] [--bw_method BW_METHOD]
                                            [--max_dist MAX_DIST] [--neighbors NEIGHBORS]
                                            [--scaling_factor SCALING_FACTOR]
                                            [--measure_from {center,right,left,top,bottom}] [--ext EXT]
                                            [--ext_stomata EXT_STOMATA] [--new_ext NEW_EXT] [--img_ext IMG_EXT]
                                            [--no_visualize] [--min_stomata MIN_STOMATA]
                                            dir

positional arguments:
  dir                   Base directory

options:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity (default: False)
  --draw_lines          Draw lines on image and save as output (default: True)
  --nlines NLINES       Number of lines to draw on image (default: 10)
  --bins BINS           Number of bins to use in angles histogram (default: 500)
  --kernel_size KERNEL_SIZE
                        Size of Sobel kernel to find direction (default: 7)
  --bw_method BW_METHOD
                        Gaussan kde parameter, controls the smoothness of curve-fit (default: 0.1)
  --max_dist MAX_DIST   Maximum distance to include (default: 1000.0)
  --neighbors NEIGHBORS
                        Number of neighbors (default: 2)
  --scaling_factor SCALING_FACTOR
                        Scaling factor, higher values follow image directionality more (default: 10.0)
  --measure_from {center,right,left,top,bottom}
                        Whether to start measuring at right edge of box (default: right)
  --ext EXT             Filename extension of masks (default: _cp_masks_filtered.png)
  --ext_stomata EXT_STOMATA
                        Filename extension of masks (default: _cp_masks_stomata.png)
  --new_ext NEW_EXT     Filename extension of masks filtered by stomata (default: _by_stomata.png)
  --img_ext IMG_EXT     Filename extension of output visualization (default: _vis.png)
  --no_visualize        Include output visualization image (default: False)
  --min_stomata MIN_STOMATA
                        Line must have at least this many stomata to be filtered (default: 3)
```


## Curve fitting

Fit a sigmoid curve to cell size against cell location

```console
$ python curve_fitting.py --help
usage: curve_fitting.py [-h] [-v] [-e EXT] [-n NEW_EXT] [-p PLOT_EXT] dir

positional arguments:
  dir                   Base directory

options:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity (default: False)
  -e EXT, --ext EXT     Filename extension of input csv (default: _in.csv)
  -n NEW_EXT, --new_ext NEW_EXT
                        Filename extension of output csv (default: _params.csv)
  -p PLOT_EXT, --plot_ext PLOT_EXT
                        Filename extension of output plots (default: _plot.png)
```
