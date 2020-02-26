import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max



def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=230):
    """
    Hough transform for lines
    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    lines_are_white - boolean indicating whether lines to be detected are white
    value_threshold - Pixel values above or below the value_threshold are edges
    Returns:
    accumulator - 2D array of the hough transform accumulator
    theta - array of angles used in computation, in radians.
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image.
    """
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)


    # Vote in the hough accumulator
    print(len(x_idxs))
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        print(i)
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('001109.png')

    if img.ndim == 3:
        img_g = rgb2gray(img)

    accumulator, thetas, rhos = hough_line(img_g)
    # show_hough_line(img, accumulator,thetas, rhos)
    #show_hough_line(img, accumulator, thetas, rhos, save_path='test.png')
    max=np.max(accumulator)

    # Comparison between image_max and im to find the coordinates of local maxima

    coordinates =peak_local_max(accumulator,min_distance=5,threshold_abs =0.8*max)
    #i, j = np.where(accumulator > max*0.8)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(accumulator, cmap='jet')
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    a = np.cos(thetas)
    b=np.sin(thetas)
    for l in range(len(coordinates)):
        index_rho= coordinates[l][0]
        index_thetas=coordinates[l][1]
        x0=a[index_thetas]*rhos[index_rho]
        y0=b[index_thetas]*rhos[index_rho]
        x1 =int(x0 + 1000*(-b[index_thetas]))
        y1 =int(y0 + 1000*a[index_thetas])
        x2 =int(x0 - 1000*(-b[index_thetas]))
        y2 = int(y0 - 1000*a[index_thetas])



        cv2.line(img, (int(x1), y1), (x2, int(y2)), (255, 0, 0), 5)
    cv2.imwrite('test.png', img)