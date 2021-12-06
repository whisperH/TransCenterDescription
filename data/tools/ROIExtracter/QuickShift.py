import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries


if __name__ == "__main__":

    img = cv2.imread("D:\\data\\dataset\\track\\0.2_0.4\\tracker3_rest6_000455.PNG")
    image = cv2.pyrDown(img)
    img = cv2.pyrUp(image)
    segments_quick = quickshift(img, kernel_size=3, max_dist=9, ratio=0.5)

    print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")

    plt.imshow(mark_boundaries(img, segments_quick))
    plt.show()
