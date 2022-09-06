import os.path
import struct
import numpy as np

MNIST_DIR = "mnist"
TRAIN_DATA = "train-images-idx3-ubyte.gz"
TRAIN_LABEL = "train-labels-idx1-ubyte.gz"
TEST_DATA = "t10k-images-idx3-ubyte.gz"
TEST_LABEL = "t10k-labels-idx1-ubyte.gz"


class DataLoader(object):
    train_data = None
    test_data = None

    def load_mnist(self, file_dir, is_images=True, debug=False):
        print("file_dir:", file_dir)
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        if is_images: #读取图像数据
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else: #load label data
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        data_size = num_images*num_rows*num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows*num_cols])
        # if debug:
        print("file_dir:", file_dir, "mat_data size:", mat_data.size, " mat_data shape:", mat_data.shape)
        return mat_data

    def load_data(self):
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)



if __name__ == '__main__':
    # print(MNIST_DIR)
    mnistLoader = DataLoader()
    mnistLoader.load_data()
    print("train:", mnistLoader.train_data.shape)
    print("test:", mnistLoader.test_data.shape)



























