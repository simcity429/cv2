import sys, os
import numpy as np
import cv2

TRAIN_PATH = "faces_training"
TEST_PATH = "faces_test"
STUDENT_CODE = '2014147550'
OUTPUT_TEXT_PATH = STUDENT_CODE + "\\output.txt"


def load_imgs(paths, flags=cv2.IMREAD_GRAYSCALE):
    imgs = []
    for i, path in enumerate(paths):
        imgs.append(cv2.imread(path, flags))
    return np.asarray(imgs)


def svd(x, flag=False):
    n, row, col = x.shape
    res_x = np.array(x, dtype="float")  # (39, 96, 84)
    res_x = np.reshape(res_x, (n, row * col))  # img-to-vec, e.g. (39, 8064) => (n, p)

    avg_x = np.mean(res_x, axis=0, keepdims=True)  # axis=0 (1, 8064)
    if flag:#returning only average; not used
        return avg_x
    res_x -= avg_x  # zero mean

    U, S, Vt = np.linalg.svd(res_x, full_matrices=flag)  # compute minimal SVD of X, e.g. (39, 39) (39,) (39, 8064)
    return avg_x, U, S, Vt


def select_pcs(S, per):
    S = S ** 2 / (len(S) - 1)
    sum_variance = 0.
    raw_variance = np.sum(S)

    for dim, val in enumerate(S, start=1):
        sum_variance += val
        if (sum_variance / raw_variance) >= per:
            return dim
    return len(S)  # input var_per > 1.0


def calculate_mse(img1, img2):
    i, j = img1.shape
    return np.sum(np.square(img1 - img2)) / (i * j)


def step1(x, per):
    avg_x, U, S, Vt = svd(x)  # res = (res_x, avg_x)
    dim = select_pcs(S, per)

    with open(OUTPUT_TEXT_PATH, "w") as f:
        f.write("##########  STEP 1  ##########\n")
        f.write("Input Percentage: %s\n" % per)  # per : float value (0, 1)
        f.write("Selected Dimension: %d\n" % dim)  # 1 ~ n

    return avg_x, U, S, Vt, dim


def step2(x, avg_x, U, S, Vt, k):
    n, row, col = x.shape
    reconstructed = (U[:, :k].dot(np.diag(S)[:k, :k])).dot(Vt[:k, :]) + avg_x#reconstruction
    reconst_errs = []
    for i in range(n):  # using the selected number of PCs
        tmp = reconstructed[i, :]
        tmp = tmp.reshape(row, col)
        reconst_errs.append(calculate_mse(x[i], tmp))
        cv2.imwrite(save_path[i], tmp)

    znum = len(str(n))
    with open(OUTPUT_TEXT_PATH, "a") as f:
        f.write("\n##########  STEP 2  ##########\n")
        f.write("Reconstruction error\n")
        f.write("average : %.4f\n" % (sum(reconst_errs) / len(reconst_errs)))
        for i, err in enumerate(reconst_errs, start=1):
            f.write(("%d" % i).zfill(znum) + ": %.4f\n" % err)
    return reconstructed


def step3(test_x, x, k, avg_x, Vt):
    data_num, _, _ = x.shape
    x = x.reshape((data_num, -1)) - avg_x
    V = Vt.T#projection matrix
    reduced_data = np.dot(x, V[:, :k])#projected data

    test_num, row, col = test_x.shape
    test_x = test_x.reshape((test_num, -1)) - avg_x
    reduced_test = np.dot(test_x, V[:, :k])#projected test images

    outputs = []
    for i in range(test_num):
        test_tmp = np.copy(reduced_test[i, :]).reshape(1, -1)
        data_tmp = np.copy(reduced_data)
        l2 = np.sqrt(np.sum((data_tmp - test_tmp) ** 2, axis=1))  # Euclidean Distance computed; shape=> (39,)
        outputs.append(np.argmin(l2))#choose argmin

    with open(OUTPUT_TEXT_PATH, "a") as f:
        f.write("\n##########  STEP 3  ##########\n")
        for i, output in enumerate(outputs):
            f.write(test_files_name[i] + " ==> " + train_files_name[output] + "\n")


if __name__ == "__main__":

    var_per = float(sys.argv[1])
    train_files_name = os.listdir(TRAIN_PATH)
    test_files_name = os.listdir(TEST_PATH)
    training_files = [os.path.join(TRAIN_PATH, x) for x in train_files_name]
    test_files = [os.path.join(TEST_PATH, x) for x in test_files_name]
    save_path = [os.path.join(STUDENT_CODE, x) for x in train_files_name]
    if not os.path.exists(STUDENT_CODE):
        os.mkdir(STUDENT_CODE)

    x = load_imgs(training_files)
    test_x = load_imgs(test_files)

    avg_x, U, S, Vt, k = step1(x, var_per)
    step2(x, avg_x, U, S, Vt, k)
    step3(test_x, x, k, avg_x, Vt)