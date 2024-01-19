import torchvision
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import numpy

from PIL import Image, ImageFont, ImageDraw
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def get_sam

sam_checkpoint = "vit_b_lm.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)



"""
Basically -- find height of image --> turn img into pix and seperate px into batches each a fraction of img height
--> turn batches into images run them through sam --> turn results back into pixles
and concatanate together into original image height """
def img_dim(filename):
    img = Image.open(filename)
    # pixels = img.load()
    width, height = img.size #assigns width and height values for iteration later
    return (width, height)

def img_to_pix(filename):
    """
    Takes a filename (must be inputted as a string with proper file attachment
    ex: .jpg, .png) and converts that image file to a list of its pixels.

    For RGB images, each pixel is a tuple of (R,G,B) values.
    For BW images, each pixel is an integer.

    Note: Don't worry about determining if an image is RGB or BW.
          The PIL library functions you use will return the correct pixel
          values for either image mode.

    Inputs:
        filename: string representing an image file, such as 'img_name.jpg'

    Returns: list of pixel values
             in form (R,G,B) such as [(0,20,55), (30,50,90), ...] for RGB image
             in form L such as [60, 66, 72...] for BW image
    """
    img = Image.open(filename)
    # pixels = img.load()
    width, height = img.size #assigns width and height values for iteration later
    list_pixles = []
    for ycord in range(height): #iterates through each pixel in a single vertical column, iterates through columns
        for xcord in range(width):
            list_pixles.append(img.getpixel((xcord, ycord)))
    return list_pixles

def crop_img(pix, dim):
    width = dim[0]
    height = dim[1]
    raw_pix_list = pix
    layer_height = int(height/5)
    xtra_layer = height % 5
    cropped_pix_batches = []
    for layers in range(1,6):
        layer_height_end = layer_height
        if layers == 5:
            layer_height_end = layer_height + xtra_layer
        cropped_pix_batches.append(raw_pix_list[(layers-1)*layer_height*width:layers*layer_height_end*width])
    return [cropped_pix_batches, layer_height, xtra_layer]

def pix_to_img(pixels, size, mode):
    """
    Creates an Image object from a inputted set of RGB tuples.

    Hint:
        Step 1: Create a new image object with the specified size and model
        Step 2: Populate the image object with the pixels. Search for putdata()!

    Inputs:
        pixels: a list of pixels such as the output of `img_to_pixels`
        size: a tuple of (width, height) representing the dimensions of the
              desired image. Assume that size is a valid input such that
              size[0] * size[1] == len(pixels).
        mode: 'RGB' or 'L' to indicate an RGB image or a BW image, respectively

    Returns:
        img: Image object made from list of pixels
    """
    new_img = Image.new(mode, size) #initalizes new image
    new_img.putdata(pixels) #loads pixel data into new image
    return new_img

filename = "2021-02-19-tris_minimal-0307_2023-05-22_068.jpg"
crop_data = crop_img(img_to_pix(filename), img_dim(filename))
pix_batches = crop_data.pop(0)
new_height = crop_data.pop(0)
xtra_height = crop_data.pop(0)
orig_dim = img_dim(filename)
new_width = orig_dim[0]
counter = 0
crop_names = []
for batches in pix_batches:
    counter = counter + 1
    if counter == len(pix_batches):
        new_height = new_height + xtra_height
    new_dim = (new_width, new_height)
    img = pix_to_img(batches, new_dim, 'RGB')
    img.save(f"{counter} crop {filename}")
    crop_names.append(f"{counter} crop {filename}")

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=60,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.4,
    crop_n_layers=2,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

from matplotlib import pyplot as plt
mask_names = []
for i in range(len(crop_names)):
  image = cv2.imread(crop_names[i])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  masks2 = mask_generator_2.generate(image)
  plt.figure(figsize=(20, 20))
  plt.imshow(image)
  show_anns(masks2)
  plt.axis('off')
  plt.savefig(f"masked_{crop_names[i]}", bbox_inches='tight', pad_inches=0)
  mask_names.append(f"masked_{crop_names[i]}")
  plt.show()

from PIL import Image, ImageFont, ImageDraw
import numpy

"""Basically
Dissolve all images into pixles --> concatenate pixels --> pix to image"""

def img_dim(filename):
    img = Image.open(filename)
    # pixels = img.load()
    width, height = img.size #assigns width and height values for iteration later
    return width, height

def img_to_pix(filename):
    """
    Takes a filename (must be inputted as a string with proper file attachment
    ex: .jpg, .png) and converts that image file to a list of its pixels.

    For RGB images, each pixel is a tuple of (R,G,B) values.
    For BW images, each pixel is an integer.

    Note: Don't worry about determining if an image is RGB or BW.
          The PIL library functions you use will return the correct pixel
          values for either image mode.

    Inputs:
        filename: string representing an image file, such as 'img_name.jpg'

    Returns: list of pixel values
             in form (R,G,B) such as [(0,20,55), (30,50,90), ...] for RGB image
             in form L such as [60, 66, 72...] for BW image
    """
    img = Image.open(filename)
    # pixels = img.load()
    width, height = img.size #assigns width and height values for iteration later
    list_pixles = []
    for ycord in range(height): #iterates through each pixel in a single vertical column, iterates through columns
        for xcord in range(width):
            list_pixles.append(img.getpixel((xcord, ycord)))
    return list_pixles

def pix_to_img(pixels, size, mode):
    """
    Creates an Image object from a inputted set of RGB tuples.

    Hint:
        Step 1: Create a new image object with the specified size and model
        Step 2: Populate the image object with the pixels. Search for putdata()!

    Inputs:
        pixels: a list of pixels such as the output of `img_to_pixels`
        size: a tuple of (width, height) representing the dimensions of the
              desired image. Assume that size is a valid input such that
              size[0] * size[1] == len(pixels).
        mode: 'RGB' or 'L' to indicate an RGB image or a BW image, respectively

    Returns:
        img: Image object made from list of pixels
    """
    new_img = Image.new(mode, size) #initalizes new image
    new_img.putdata(pixels) #loads pixel data into new image
    return new_img

number_images = len(mask_names)
pix = []
width = 0
height = 0
for i in range(number_images):
    filename = mask_names[i]
    pix2add = img_to_pix(filename)
    for pixles in pix2add:
        pix.append(pixles)
    width2add, height2add = img_dim(filename)
    width = width2add
    height = height + height2add
img = pix_to_img(pix, (width, height), 'RGB')
img.save(f"forSAM_{filename}")
img.show()

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.9,
    crop_n_layers=2,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

from matplotlib import pyplot as plt
image = cv2.imread(f"forSAM_{filename}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks2 = mask_generator_2.generate(image)
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.savefig(f"SAM_{filename}", bbox_inches='tight', pad_inches=0)
plt.show()
