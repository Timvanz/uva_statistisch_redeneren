import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm


N = 256
M = 25


def sorted_eig(mat):
    """
    Calculate all eigen values in order.
    """

    d, U = np.linalg.eig(mat)
    si = np.argsort(d)[-1::-1]
    d = d[si]
    U = U[:, si]

    return d, U


def compare_details(img, pos, d, U, mean_vec, k):
    x, y = pos

    detail = img[x:x+M, y:y+M].reshape((M*M, 1))
    df_detail = detail - mean_vec

    # Calculate coordinates with respect to eigenvector basis.
    yzm = U.T.dot(df_detail)

    # Reconstruct df_detail as linear combination of the first k eigenvectors.
    df_detail_k = U[:, :k].dot(yzm[:k])
    return df_detail_k + mean_vec


def main():
    img = imread('trui.png')

    try:
        d = np.load('d.npy')
        U = np.load('U.npy')
        mean_vec = np.load('mean_vec.npy')
    except IOError:
        sum_mat = np.zeros((M*M, M*M), np.int)
        sum_vec = np.zeros((M*M, 1), np.int)

        for i in range(N-M+1):
            for j in range(N-M+1):
                detail = img[i:i+M, j:j+M].reshape((M*M, 1))

                sum_mat += detail * detail.T
                sum_vec += detail

        mean_vec = sum_vec / float(N*N)
        mean_mat = N * mean_vec * mean_vec.T

        S = (sum_mat - mean_mat) / ((M*M)-1)

        d, U = sorted_eig(S)
        np.save('d', d)
        np.save('U', U)
        np.save('mean_vec', mean_vec)

    plt.yscale('log')
    plt.ylabel('$\lambda$-value')
    plt.xlabel('number of eigen values')
    plt.bar(range(10), d[:10], log=True)
    plt.savefig('docs/eigenvalues.png')
    plt.show()

    for k in [0, 1, 3, 5, 7, 10, 20, 30, 50, 100, 150, 200, 300,
              400, 500, 550, 600, 625]:
        eig_img = np.zeros((N, N))

        # Reconstruct whole image as local structures of N x N
        for i in range(0, 232, 25):
            for j in range(0, 232, 25):
                subimg = compare_details(img, (i, j), d, U, mean_vec, k)
                eig_img[i:i+25, j:j+25] = subimg.reshape((25, 25))

        plt.imshow(eig_img, cmap=cm.Greys_r)
        plt.title('number of eigenvalues used for reconstruction\n k = ' +
                  str(k))
        plt.savefig('docs/k'+str(k)+'.png')
        plt.show()

if __name__ == '__main__':
    main()
