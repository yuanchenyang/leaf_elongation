### **Leaf Elongation Image Analysis Instructions**

### Train a model

1. Corp images (so that each image has \~30 cells to segment) and manually
   segment cells using Cellpose GUI to generate training examples (see below for
   examples). Usually 3-5 manually segmented images (each with \~30 marked
   cells) are sufficient.

![][image1]![][image2]

2. Finetune Cellpose base model on these training images to find the best ML
   parameters that work for your images.

```python
python -m cellpose \
   	--verbose \
   	--use_gpu \
   	--train \
   	--dir train \
   	--pretrained_model cyto \
   	--chan 0 \
   	--n_epochs 8000 \
   	--learning_rate 0.04 \
   	--weight_decay 0.0001 \
   	--nimg_per_epoch 8 \
   	--min_train_masks 1 \
   	--mask_filter '_cp_masks' \
&& mv train/models/* train/models/CP_model
```

3. Most important parameters to tune (see [documentation](https://cellpose.readthedocs.io/en/latest/cli.html) for more details):
   * `pretrained_model`: Base model to finetune, cyto works best for our images
   * `chan`: Color channel(s) to train on
   * `learning_rate`: A larger value trains faster but too large can lead to
     instabilities, so choose this as large as possible. Try to vary this on a
     log-scale.
   * `n_epochs`: The number of training steps. Increase until the training loss
     stops decreasing and stabilizes. We find that the model produces good
     results when the final training loss is around \~0.3, usually achieved with
     1-2k epochs.
4. Test the model on new images. This will generate plots displaying the segmented cells, as well as `.png` files that can be fed into downstream pipeline stages.

```python
python -m cellpose \
   	--verbose \
   	--use_gpu \
   	--dir test \
   	--pretrained_model train/models/CP_model \
   	--diameter 0. \
   	--chan 0 \
   	--flow_threshold 0.6 \
   	--cellprob_threshold '-0.5' \
   	--save_mpl \
   	--save_png \
   	--save_txt \
   	--no_npy \
   	--savedir test
```

![][image3]

5. Most important parameters to tune:
   * `chan`: Set to same channel as used in training
   * `flow_threshold`: Increase to increase number of masks at the expense of
     mask quality
   * `cellprob_threshold`: Decrease to find larger masks at the expense of
      false positives
6. Check the test output. If there are images where the model produces a lot of
   false positives/negatives, it’s probably because that image is significantly
   different from training examples. In this case, mark the cells by hand and
   add into the training set as another example.
7. Repeat the above until the model is sufficiently accurate then save for later
   usage. Note: the training can be done on any features, e.g., on one type of
   cells, one multi-cellular structure if the ML can faithfully differentiate
   it. For example, we used sister cells in this study given it is one type of
   cell that can be located next to the stomata which gives this type of cells a
   unique shade than other types of cells. Ideally if you can train and
   reproduce the cells in the training set, it should be able to recognize them
   in a new test set.

### Run models on one sample (Barley-Domesticated-Drought-5) in Barley samples

1. Input: images folder of one sample, and 2 models that each work best for
   different parts of the sample (e.g. undifferentiated cells and sister cells)
2. Run cellpose using each of the 2 models to segment
   ![][image4]
   Undifferentiated cells
   ![][image5]
   Sister cells at small cell region
   ![][image6]
   Sister cells at mature cell region
3. Quality check: similar to what we did to get an accurate model. If the cell
   segmentations are not accurate, go back to the model training step to get
   more accurate models. However, because the cell length changes gradually
   along the leaf axis, and each cell file going in parallel should change
   similarly along the leaf axis, the cells recognized spatially nearby can be
   used as replicates. This increases the accuracy of the trend fitting along
   the leaf axis in the next step.
4. Extract information of cells’ relative positions and diameters along leaf
   a. Automatically stitching adjacent images to compute relative offsets using stitching.py
      ![][image7]
      ![][image8]
   2. Find directionality of cells in image to approximate leaf as a piecewise linear function using directionality.py
      ![][image9]
   3. First filter out spurious segmented cells using filter\_mask\_png.py, then extract cells’ coordinates and diameter with cell\_diameter.py
   4. The following R script is used to extract cell length and absolute locations along the leaf for a single model’s output. Cell lengths are grouped then averaged in bins of width 300𝞵m and cell length is only output if there is more than 20 cells in the bin. Parameters requiring manual input are directionality of the leaf image orientation (option of going left or right along the images).
      *numbercut1==20*
      *step==300*
      *orientation\[oo\]==”right”*
      *\#read in sister cell length(pixel has a ratio of 2.7 with real um and 3826 as total image length)*
      *cell00 \<-read.csv(paste0("image\_results/barley\_all/",list2\[oo\],"/cells.csv"), header\=T)*
      *head(cell00)*
      *cell00\[,3\]\<-cell00\[,3\]/2.7*
      *cell00\[,4\]\<-cell00\[,4\]/2.7*
      *cell00\[,5\]\<-rowMaxs(cbind(cell00\[,5\]/2.7, cell00\[,6\]/2.7))*
      *\#\#read in image relative locations*
      *location \<-read.csv(paste0("images/leaf\\ extension\\ 2/",list1\[oo\],"/stitching\_result.csv"), header\=T)*
      *dim(location)*
      *head(location)*
      *location\[,3\]\<-location\[,3\]/2.7*
      *location\[,4\]\<-location\[,4\]/2.7*
      *\#\#get cell locations along leaf axis*
      *real\_location\<-rep(0,dim(location)\[1\])*
      *real\_location\[1\]\<-location$X.offset\[1\]*
      *for (i in 2:dim(location)\[1\]){*
      		*real\_location\[i\]\<-sum((location$X.offset\[1:i\]))*
      *}*
      *real\_location \<-real\_location\[match(substr(cell00$Name,1,list4\[oo\]),substr(location\[,2\],1,list4\[oo\]))\]*
      *real\_location\[which(substr(cell00$Name,1,list4\[oo\])==substr(location\[1,1\],1,list4\[oo\]))\] \<-0*
      *if(orientation\[oo\]=="right"){*
      	*\#going right*
      *cell00\[,8\] \<-c(cell00\[,3\]/cos(direction\[match(substr(cell00\[,1\],1,namelength\[oo\]),substr(direction\[,1\],1,namelength\[oo\]) ),2\])+ real\_location)-emptylength-divisionzone*
      	*}else{*
      *\#going left*
      	*cell00\[,8\] \<- c((3826/2.7\-cell00\[,3\])/cos(direction\[match(substr(cell00\[,1\],1,namelength\[oo\]),substr(direction\[,1\],1,namelength\[oo\]) ),2\])- real\_location)-emptylength-divisionzone*
      	*}*
      *\#get average within each 300um bin*
      *window\<-(cell00\[,8\]) %/% step*
      *df \<-as.data.frame(cbind(cell00$Width, window))*
      *cell00\_data\<-as.matrix(df %\>% group\_by(window) %\>%*
        *summarize(res\=quantile(V1,probs\=0.5),number\=length(V1)))*
      *cell00\_data\<-cell00\_data\[which(cell00\_data\[,3\]\> numbercut1),\]*

5. Other information needed on a per-sample basis:
   1. The empty space between the beginning of the leaf and the image edge
   2. Division zone and elongation zone separation location (in this study defined as the symmetric and unsymmetric division of stomata cells or trichome cells, and this is hand marked in images and read in directly)
   3. In a small number of images that are hard to get accurate cell segmentations automatically, cells can be hand-marked and read in directly, though including these images has a negligible effect on the final curve fitting
6. Aggregate outputs of 2 models. These two models should have gradual changes and overlap at the common region except the region when one model works obviously better than another model. If not, retrain the models to better accuracy.
   ![][image10]
7. Fit into a sigmoid curve, extracting curve parameters using curve\_fitting.py (a cutoff may be used to only fit the model in regions before cell length decrease at the tip of the leaf) In our study, R2 is higher than 0.75, with a mean of 0.92.
   ![][image11]

### Examples of variations

1. Using trichome to identify the cell lengths in between. First train a model to recognize trichomes. Then measure distance along leaf direction between each pair of trichomes as a proxy for the cell length between them using cell\_pairwise\_dist.py. Usually a lower quantile is used to extract mean cell length in each bin, since it tends to be bigger than the real value affected by missing trichomes detected.
   ![][image12]
2. Find sister cells in Oat when there are two files of stomata cells next to each other, by training an additional model to recognize stomata cells, and use them to more accurately identify sister cells with filter\_sister\_cells\_using\_stomata.py.
   ![][image13]

### Useful tool list to assemble pipelines

For more details, see [`scripts/readme.md`](/scripts/readme.md).

1. Image stitching and extract coordination between two neighboring images

   `stitching.py`

2. Image merging

   `merge_imgs.py`

3. Image directionality and extract distribution

   `directionality.py`

4. Extract ROI coordinates and ferret’s diameters

   `cell_diameter.py`

5. Cell pairwise distance along image directionality

   `cell_pairwise_dist.py`

6. Filter masks that are noise (e.g., areas smaller than 1, usually used before used as training set)

   `filter_mask_png.py`

7. Shrink images based on their width (this is usually needed if the cells are too long that can not be recognized by Cellpose)

   `shrink_width.py`

8. Visualize masks

   `visualize_masks.py`

9. Filter sister cells using stomata

   `filter_sister_cells_using_stomata.py`

10. Fit a sigmoid curve to cell size against cell location

    `curve_fitting.py`

(input data format: first column with name location; other columns as samples with name of the sames as colnames. Can input a vector of cutoff for tails otherwise use the whole column for fitting. Output parameters in a csv, and output the fitting curve and original data in figures in a pdf. Output rownames are sample names, and colnames are parameters, add l10\_90)
\* all the python instruction can be found using {function} \-help to get default parameters and flags
