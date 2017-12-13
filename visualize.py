import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# -------------------------------------------------------------------------

def plot_images(A, show_p=False, filepath=None, title=None):
    """
    Matrix A represents multiple square images, each "unrolled" into a
    column vector.  This function reshapes each column as a square array
    and displays all of the images in an approximately square grid.
    By default the function will plot the grid as a matplotlib image.
    If filepath is specified, the image will be save to that filepath.
    You should not edit this function.
    Additional parameters are specified within the function, but you will not
    use them.
    opt_normalize: whether we need to normalize the filter so that all of
    them can have similar contrast. Default value is true.
    opt_graycolor: whether we use gray as the heat map. Default is true.
    :param A: matrix: col = image, rows = pixels in image
    :param show_p: boolean for whether to plot the grid as a matplotlib plot.
    :param filepath: path to save image
    :param title: optional string to display in plot title
    :return:
    """
    opt_normalize = True
    opt_graycolor = True

    # Rescale
    A = A - np.average(A)

    # Compute rows & cols
    (row, col) = A.shape
    sz = int(np.ceil(np.sqrt(row)))
    buf = 1
    n = np.ceil(np.sqrt(col))
    m = np.ceil(col / n)

    # (buf + m * (sz + buf), buf + n * (sz + buf)))
    image = np.ones(shape=(int(buf + m * (sz + buf)), int(buf + n * (sz + buf))))

    if not opt_graycolor:
        image *= 0.1

    k = 0
    for i in range(int(m)):
        for j in range(int(n)):
            if k >= col:
                continue

            clim = np.max(np.abs(A[:, k]))

            if opt_normalize:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz,
                      buf + j * (sz + buf):buf + j * (sz + buf) + sz] \
                    = A[:, k].reshape(sz, sz) / clim
            else:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz,
                      buf + j * (sz + buf):buf + j * (sz + buf) + sz] \
                    = A[:, k].reshape(sz, sz) / np.max(np.abs(A))
            k += 1

    if title:
        plt.title(title)

    if show_p or filepath is None:
        plt.imshow(image, cmap=matplotlib.cm.gray)
        plt.show()

    if filepath is not None:
        plt.imsave(filepath, image, cmap=matplotlib.cm.gray)


# -------------------------------------------------------------------------
