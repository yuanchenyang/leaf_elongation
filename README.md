## Remove small masks(usually noise) of mask like files in the folder, check help to change pattern of files

```
python filter_mask_png.py /Users/jiey/Downloads/base_model/test
```

## Get cell sizes, similar idea, check help for details

```
python cell_diameter.py /Users/jiey/Downloads/base_model/test
```

## Stitch to get the relative locations of sequential two images in the folder, check help for details

```
python stitching.py /Users/jiey/Downloads/OneDrive-2023-07-24/Wheat Z W 1 control
```

## Measure two trichome(or any ROI) distances, can change scaling factors to make sure along leaf axis, can also check results and adjust parameters

```
python cell_pairwise_dist.py --scaling_factor 50 /Users/jiey/Downloads/trichome_model/test3
```
