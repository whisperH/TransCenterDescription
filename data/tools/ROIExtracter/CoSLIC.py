# the code forked from https://github.com/SamMaoYS/Region-Growing-Segmentation/blob/b8ab945c347e7cf97cec8ea576d1340982d872d3/src/coslic.py
import cv2
import matplotlib.pyplot as plt
from cv2 import ximgproc
from heatmap import _add_instance
import numpy as np
def gaussian_blur(img, sigma=np.sqrt(2), size=-1):
    k_size = size
    if k_size == -1:
        k_size = np.int(2 * np.ceil(4 * sigma) + 1)
    blur = cv2.GaussianBlur(img, (k_size, k_size), sigma)
    return blur



class CoSLIC:
    def __init__(self, img, s=40, compactness=0.01, max_iter=20):
        self._img = img
        self._compactness = compactness
        self._max_iter = max_iter
        self._s = s
        self._labels = None
        self._edges = None
        self._n_labels = 0

    def slic(self):
        blur = gaussian_blur(self._img, sigma=0, size=3)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        slic = ximgproc.createSuperpixelSLIC(
            lab, algorithm=ximgproc.SLIC,
            region_size=self._s, ruler=self._compactness
        )
        slic.iterate(num_iterations=self._max_iter)
        slic.enforceLabelConnectivity(min_element_size=self._s)
        mask = slic.getLabelContourMask(thick_line=False)
        self._labels = slic.getLabels()
        self._n_labels = slic.getNumberOfSuperpixels()
        mask_c3 = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        slic_viz = (mask_c3 == 0) * self._img
        mask_c3[:, :, 0:2] = 0
        slic_viz = slic_viz + mask_c3
        return slic_viz

    def canny_edge(self, low=0.1, high=0.2):
        blur = gaussian_blur(self._img, 0, 5)
        dx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
        dy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
        gradient = np.zeros_like(dx)
        cv2.magnitude(dx, dy, gradient)
        max_grad = np.amax(gradient)
        # print(max_grad)
        return cv2.Canny(np.int16(dx), np.int16(dy), max_grad * low, max_grad * high)

    def fill(self, data, i, j, target, val):
        if data[i, j] == val or data[i, j] != target:
            return
        data[i, j] = val
        q = [(i, j)]
        h, w = np.shape(data)
        while q:
            n = q.pop(0)
            if n[1] > 0 and data[n[0], n[1]-1] == target:
                idx_i = n[0]
                idx_j = n[1]-1
                data[idx_i, idx_j] = val
                q.append((idx_i, idx_j))
            if n[1]+1 < w and data[n[0], n[1]+1] == target:
                idx_i = n[0]
                idx_j = n[1]+1
                data[idx_i, idx_j] = val
                q.append((idx_i, idx_j))
            if n[0] > 0 and data[n[0]-1, n[1]] == target:
                idx_i = n[0]-1
                idx_j = n[1]
                data[idx_i, idx_j] = val
                q.append((idx_i, idx_j))
            if n[0]+1 < h and data[n[0]+1, n[1]] == target:
                idx_i = n[0]+1
                idx_j = n[1]
                data[idx_i, idx_j] = val
                q.append((idx_i, idx_j))
            # cv2.imshow("data", data)
            # cv2.waitKey(0)

    def flood_fill(self):
        print(np.shape(self._img))
        self._edges = self.canny_edge()
        new_label = 1
        result = np.zeros_like(self._labels, np.uint16)
        target = np.iinfo(np.uint16).max

        for label in range(self._n_labels):
            l_x, l_y = np.where(self._labels == label)
            e_x, e_y = np.where(self._edges == 255)
            data = np.zeros_like(self._labels, np.uint16)
            data[l_x, l_y] = target
            data[e_x, e_y] = 0

            while True:
                idx = next((idx for idx, x in np.ndenumerate(data) if x == target), (None, None))
                print(idx)
                if idx != (None, None):
                    self.fill(data, idx[0], idx[1], target, new_label)
                    new_label += 1
                else:
                    break

            result += data

        r = np.random.choice(255, new_label-1)
        g = np.random.choice(255, new_label-1)
        b = np.random.choice(255, new_label-1)

        show = np.zeros_like(self._img)
        for i in range(1, new_label, 1):
            show[result == i] = [r[i-1], g[i-1], b[i-1]]
        return show


if __name__ == '__main__':
    from skimage import io
    picturepath = "D:\\data\\dataset\\track\\0.2_0.4\\tracker5_cruise1_000082.PNG"
    # picturepath = "E:\\code\\TransCenter\\data\\config\\template_straight.PNG"
    # picturepath = "E:\\code\\TransCenter\\data\\config\\template_turn.PNG"
    img = cv2.imread(picturepath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heat_map = _add_instance(img)
    slic = CoSLIC(img)
    slic_viz = slic.slic()
    # edges = slic.canny_edge()
    flood = slic.flood_fill()

    plt.imshow(flood)
    plt.show()