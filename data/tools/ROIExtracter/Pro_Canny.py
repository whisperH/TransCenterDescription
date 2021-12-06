from scipy import ndimage as ndi
from skimage.feature import canny
from matplotlib import pyplot as plt
import numpy as np

def CANNY(img, hyp_config=None):
    edges = canny(img)
    fill_coins = ndi.binary_fill_holes(edges)
    # label_objects, nb_labels = ndi.label(fill_coins)
    # sizes = np.bincount(label_objects.ravel())
    # mask_sizes = sizes > 10
    # mask_sizes[0] = 0
    # coins_cleaned = mask_sizes[label_objects]
    plt.imshow(fill_coins)
    plt.show()
if __name__ == '__main__':
    from skimage import io
    picturepath = "D:\\data\\dataset\\track\\0_0.1\\tracker13_cruise1_000109.PNG"
    img = io.imread(picturepath, as_gray=True)

    CANNY(img)


