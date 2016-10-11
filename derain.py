#coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cv2

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version

#展示函数
def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    plt.figure(figsize=(5, 3.3))
    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(image, 'gray')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 3, 3)
    difference = image - reference

    plt.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, 'gray')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 3, 2)
    plt.title('origin_image')
    plt.imshow(reference, 'gray')
    plt.xticks(())
    plt.yticks(())

    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)




#初始化的配置
origin_image = ['0001a.png', '1000.png', '0026c.png', '0027c.png']
noise_image = ['0001.png', '0000.png', '0026b.png', '0027b.png']

patch_size = (3, 3)


for i in range(4):
    img1 = cv2.imread(origin_image[i], 0)
    img2 = cv2.imread(noise_image[i], 0)

    #img1 = img1/255.0

    #训练集图片切块，归一化预处理
    data = extract_patches_2d(img1/255.0, patch_size)
    data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)

    data2 = extract_patches_2d(img2/255.0, patch_size)
    data2 = data2.reshape(data2.shape[0], -1)
    intr = np.mean(data2, axis=0)
    data2 -= intr
    std = np.std(data2, axis=0)
    data2 /= std

    print('Learning the dictionary...')
    dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
    V = dico.fit(data).components_

    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(V[:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        plt.suptitle('Dictionary learned from face patches\n',
                     fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    transform_algorithms = [
                    ('Orthogonal Matching Pursuit\n1 atom', 'omp',
                        {'transform_n_nonzero_coefs': 1}),
                    ('Orthogonal Matching Pursuit\n2 atoms', 'omp',
                        {'transform_n_nonzero_coefs': 2}),
                    ('Least-angle regression\n5 atoms', 'lars',
                        {'transform_n_nonzero_coefs': 5}),
                    ('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': .1})]
    reconstructions = {}
    for title, transform_algorithm, kwargs in transform_algorithms:
        print(title + '...')
        dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
        code = dico.transform(data2)
        patches = np.dot(code, V)
        if transform_algorithm == 'threshold':
            patches -= patches.min()
            patches /= patches.max()

        patches *= std
        patches += intr
        patches = patches.reshape(len(data2), *patch_size)
        if transform_algorithm == 'threshold':
            patches -= patches.min()
            patches /= patches.max()

        reconstructions[title] = reconstruct_from_patches_2d(patches, (img2.shape))
        show_with_diff(reconstructions[title], img2/255.0, title)
    plt.show()
