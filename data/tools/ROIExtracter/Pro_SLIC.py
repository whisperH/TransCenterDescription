from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt


def SLIC(img, hyp_config=None):
    labels1 = segmentation.slic(
        img, n_segments=6, compactness=0.01,
        max_iter=10, start_label=1,
        convert2lab=True
    )
    plt.imshow(labels1)
    plt.show()
if __name__ == '__main__':
    from skimage import io
    picturepath = "D:\\data\\dataset\\track\\0.2_0.4\\tracker3_rest6_000455.PNG"
    img = io.imread(picturepath)

    SLIC(img)


