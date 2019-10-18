import os
import cv2
import time
import imutils
import numpy as np
sift = cv2.xfeatures2d.SIFT_create()


def parse_sift_output(target_path):
    """
    Return:
        kp: keypoint of hessian affine descriptor. location, orientation etc... OpenCV KeyPoint format. 
        des: 128d uint8 np array
    """
    
    # print(os.listdir("./sample"))
    kp = []
    des = []
    with open(target_path, "r") as f:
        lines = list(map(lambda x: x.strip(), f.readlines()))
        num_descriptor = int(lines[1])
        lines = lines[2:]
        for i in range(num_descriptor):
            # print(i, lines[i])
            val = lines[i].split(" ")
            x = float(val[0])
            y = float(val[1])
            a = float(val[2])
            b = float(val[3])
            c = float(val[4])
            # TODO: generate ellipse shaped key point
            # Refer: https://math.stackexchange.com/questions/1447730/drawing-ellipse-from-eigenvalue-eigenvector
            # Refer: http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/display_features.m
            # Refer: http://www.robots.ox.ac.uk/~vgg/research/affine/detectors.html
            key_point = cv2.KeyPoint(x, y, 1)
            sift_descriptor = np.array(list(map(lambda x: int(x), val[5:])), dtype=np.uint8)
            kp.append(key_point)
            des.append(sift_descriptor)
        
    
    return kp, np.array(des)


def resize(img):
    h, w, _ = img.shape
    if w > 500:
        img = imutils.resize(img, width=500)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def hessian_sift_extractor(path1, path2):
    os.system('./hesaff_c++/hesaff {}'.format(path1))
    os.system('./hesaff_c++/hesaff {}'.format(path2))

    kp1, des1 = parse_sift_output(path1 + '.hesaff.sift')
    kp2, des2 = parse_sift_output(path2 + '.hesaff.sift')
    des1 = des1.astype('float32')
    des2 = des2.astype('float32')
    return kp1, des1, kp2, des2

def sift_extractor(img1, img2):
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    return kp1, des1, kp2, des2


def show_match(path1, path2):

    img1 = cv2.imread(path1, 1)
    img2 = cv2.imread(path2, 1)
    img1_gray  = resize(img1.copy())
    img2_gray  = resize(img2.copy())
    
    kp1, des1, kp2, des2 = hessian_sift_extractor(path1, path2)
    # kp1, des1, kp2, des2 = sift_extractor(img1_gray, img2_gray)

    #### FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50) #指定递归次数

    begin = time.time()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    

    del_same = []
    mat = 0
    for i, (m, n) in enumerate(matches):
        # ratio test as per Lowe's paper
        if m.distance < 0.6 * n.distance:
            mat += 1
            matchesMask[i] = [1, 0]
            del_same.append(m.trainIdx)
    count = len(set(del_same))
    print('匹配耗时：{:.5f}, 匹配到的点个数：{}'.format(time.time() - begin, count))




    draw_params = dict(matchColor = (0, 255, 0),
                       singlePointColor = (255, 0, 0),
                       matchesMask = matchesMask,
                       flags = 0)
    img3 = cv2.drawMatchesKnn(img1_gray, kp1, img2_gray, kp2, matches, None, **draw_params)
    # img3 = cv2.drawKeypoints(img1, kp1, img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #画出特征点，并显示为红色圆圈

    cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('img',img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('test.jpg', img3)


if __name__ == '__main__':

    path1 = './test_pics/1.jpg'
    path2 = './test_pics/2.jpg'
    show_match(path1, path2)
