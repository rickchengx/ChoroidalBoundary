import cv2
import numpy as np
import os
import math
import sys
import csv

horizontal_scale = 1500 / 120
vertical_scale = 200 / 51

class ChoroidalBoundaryDetection:
    def detect(self, path, res_path, niblack_path, binary_threshold, sba_size, r1, r2, center_width, niblack_n, niblack_k):
        img = cv2.imread(path)
        # cut original image
        img = img[10:490, 582:1182]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find BM
        img_sn = self.smoothAndNormalized(img_gray)
        ixn = self.getVerticalMultiGradient(img_sn, 5)
        bm = self.getMaxLine(ixn)
        self.colorBoundary(img, bm, (0, 255, 0))

        img_sn = self.smoothAndNormalized(img_gray)
        ix = self.getVerticalGradient(img_sn)
        iad = self.getIntensityAscendingDistance(ix)
        idd = self.getIntensityDescendingDistance(ix)

        mi = self.getMaximumIntensityImage(img_sn, idd)
        cvi = self.getChoroidalVesselImage(img_sn, mi)
        bcvi = self.getBinarizedChoroidalVesselImage(cvi, binary_threshold)

        self.removeSmallBrightArea(bcvi, sba_size)
        cbcvi = self.getCoarseBinarizedChoroidalVesselImage(bcvi, r1, r2)

        upInterface = self.getUpInterface(cbcvi)
        downInterface = self.getDownInterface(mi, upInterface, 0.5)
        csjn = self.getCSJNeighborhood(upInterface, downInterface)

        iadx = self.getIntensityAscendingDistanceDifference(iad)
        csjcost = self.getCSJCost(iadx, csjn)

        # find csj
        csj = self.getMaxLine(csjcost)
        self.colorBoundary(img, csj, (0, 0, 255))

        # width = 1500 um = 120 cols
        width = center_width / horizontal_scale
        ups, downs, left_boundary, right_boundary = self.getCenterArea(img_gray, bm, csj, width)
        self.colorBoundary(img, left_boundary, (255, 0, 0))
        self.colorBoundary(img, right_boundary, (255, 0, 0))

        # height = 200 um = 51 cols
        angle, mean_height = self.computeMeanHeight(img_gray, ups, downs)
        print(angle, mean_height)

        niblack = self.getNiBlackImage(img_gray, ups, downs, niblack_n, niblack_k)
        black_per = self.computeBlackPercentage(niblack, ups, downs)
        self.outputImage(niblack, niblack_path)
        print(black_per)

        """
        self.outputNormalizedImage(ix, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/ix.png")
        self.outputNormalizedImage(iad, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/iad.png")
        self.outputNormalizedImage(idd, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/idd.png")
        self.outputNormalizedImage(mi, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/mi.png")
        self.outputImage(img_gray, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/img_gray.png")
        self.outputNormalizedImage(bcvi, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/bcvi.png")
        self.outputNormalizedImage(cbcvi, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/cbcvi.png")
        self.outputNormalizedImage(upInterface, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/upInterface.png")
        self.outputNormalizedImage(downInterface, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/downInterface.png")
        self.outputNormalizedImage(csjn, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/csjn.png")
        self.outputNormalizedImage(iadx, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/iadx.png")
        self.outputNormalizedImage(csjcost, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/csjcost.png")
        """

        self.outputImage(img, res_path)

        return angle, mean_height, black_per

    def printElements(self, img):
        rows, cols = img.shape
        ss = set()

        for i in range(rows):
            for j in range(cols):
                ss.add(img[i][j])

        print(ss)

    def outputNormalizedImage(self, img, path):
        normalized = None
        normalized = cv2.normalize(img, normalized, 255, 0, cv2.NORM_MINMAX)
        cv2.imwrite(path, normalized,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    def outputImage(self, img, path):
        cv2.imwrite(path, img,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    def smoothAndNormalized(self, img):
        # bilateralFilter
        img_bf = cv2.bilateralFilter(img, 5, 50, 50)

        return img_bf

    def getVerticalGradient(self, img):
        rows, cols = img.shape
        ix = np.zeros(img.shape, int)

        # compute vertical gradient in the axial direction
        for j in range(cols):
            for i in range(0, rows - 1):
                ix[i][j] = int(img[i][j]) - int(img[i + 1][j])

        return ix

    def getVerticalMultiGradient(self, img, n):
        rows, cols = img.shape
        ixn = np.zeros(img.shape, int)

        # compute vertical gradient in the axial direction
        for j in range(cols):
            for i in range(n, rows - n - 1):
                above = 0
                below = 0
                for k in range(1, n + 1):
                    above += img[i - k][j]
                    below += img[i + k][j]
                ixn[i][j] = above - below

        return ixn

    def getIntensityAscendingDistance(self, ix):
        rows, cols = ix.shape
        iad = np.zeros(ix.shape, int)

        for j in range(cols):
            for i in range(1, rows):
                if ix[i - 1][j] < ix[i][j]:
                    iad[i][j] = iad[i - 1][j] + 1
                else:
                    iad[i][j] = 0

        return iad

    def getIntensityDescendingDistance(self, ix):
        rows, cols = ix.shape
        idd = np.zeros(ix.shape, int)

        for j in range(cols):
            for i in range(rows - 2, -1, -1):
                if ix[i + 1][j] < ix[i][j]:
                    idd[i][j] = idd[i + 1][j] + 1
                else:
                    idd[i][j] = 0

        return idd

    def getMaximumIntensityImage(self, img, idd):
        rows, cols = img.shape
        maxdp = np.zeros(img.shape, int)
        mi = np.zeros(img.shape, int)

        # compute the max in img using dp, maxdp[i][j] = max value through [i ~ rows-1, j]
        for j in range(cols):
            maxdp[rows - 1][j] = img[rows - 1][j]
            for i in range(rows - 2, -1, -1):
                maxdp[i][j] = max(maxdp[i + 1][j], img[i][j])

        for j in range(cols):
            for i in range(rows):
                mi[i][j] = maxdp[i + idd[i][j]][j]

        return mi

    def getChoroidalVesselImage(self, img, mi):
        rows, cols = img.shape
        cvi = np.zeros(img.shape, int)

        for i in range(rows):
            for j in range(cols):
                cvi[i][j] = mi[i][j] - img[i][j]

        return cvi

    def getBinarizedChoroidalVesselImage(self, cvi, threshold):
        rows, cols = cvi.shape
        bcvi = np.zeros(cvi.shape, np.uint8)

        for i in range(rows):
            for j in range(cols):
                if cvi[i][j] < threshold:
                    bcvi[i][j] = 0
                else:
                    bcvi[i][j] = 255

        return bcvi

    def removeSmallBrightArea(self, bcvi, sba_size):
        contours, hierarchy = cv2.findContours(bcvi, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        small_contours = []
        for t in contours:
            if cv2.contourArea(t) < sba_size:
                small_contours.append(t)

        cv2.drawContours(bcvi,small_contours, -1, 0, cv2.FILLED)

    def getCoarseBinarizedChoroidalVesselImage(self, bcvi, r1, r2):
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r1, r1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r2, r2))

        bcvi_dilate = cv2.dilate(np.uint8(bcvi), kernel1, iterations=1)
        cbcvi = cv2.erode(bcvi_dilate, kernel2, iterations=1)

        return cbcvi

    def getUpInterface(self, cbcvi):
        rows, cols = cbcvi.shape
        upInterface = np.zeros(cbcvi.shape, int)

        for j in range(cols):
            flag = True
            for i in range(rows - 1, -1, -1):
                if flag and cbcvi[i][j] == 255:
                    flag = False
                if flag:
                    upInterface[i][j] = 0
                else:
                    upInterface[i][j] = 1

        return upInterface

    def getDownInterface(self, mi, upInterface, ratio):
        rows, cols = mi.shape
        downInterface = np.zeros(mi.shape, int)

        # find the rows of upInterface
        rows_upInterface = []
        for j in range(cols):
            for i in range(rows - 1, -1, -1):
                if upInterface[i][j] == 1:
                    rows_upInterface.append(i)
                    break

        if len(rows_upInterface) == 0:
            print("Error, No up interface")

        for j in range(cols):
            for i in range(rows):
                if mi[i][j] < int((mi[rows_upInterface[j]][j] + mi[rows - 1][j]) * ratio):
                    downInterface[i][j] = 1
                else:
                    downInterface[i][j] = 0

        return downInterface

    def getCSJNeighborhood(self, upInterface, downInterface):
        rows, cols = upInterface.shape
        csjn = np.zeros(upInterface.shape, int)

        for j in range(cols):
            for i in range(rows):
                if upInterface[i][j] == 0 and downInterface[i][j] == 0:
                    csjn[i][j] = 1
                else:
                    csjn[i][j] = 0

        return csjn

    def getIntensityAscendingDistanceDifference(self, iad):
        rows, cols = iad.shape
        iadx = np.zeros(iad.shape, int)

        for j in range(cols):
            for i in range(0, rows - 1):
                iadx[i][j] = int(iad[i][j]) - int(iad[i + 1][j]) + 1

        return iadx

    def getCSJCost(self, iadx, csjn):
        rows, cols = iadx.shape
        csjcost = np.zeros(iadx.shape, int)

        for i in range(rows):
            for j in range(cols):
                csjcost[i][j] = int(iadx[i][j]) * int(csjn[i][j])

        return csjcost

    def getMaxLine(self, img):
        rows, cols = img.shape

        costs = [[0] * cols for i in range(rows)]
        last_opt_node = [[(0, 0)] * cols for i in range(rows)]

        # init
        for i in range(rows):
            costs[i][0] = int(img[i][0])
            last_opt_node[i][0] = (-1, -1)

        # dp
        for j in range(1, cols):
            for i in range(rows):
                opt_cost = costs[i][j - 1]
                opt_pos = (i, j - 1)
                if i > 0 and costs[i - 1][j - 1] > opt_cost:
                    opt_cost = costs[i - 1][j - 1]
                    opt_pos = (i - 1, j - 1)
                if i < rows - 1 and costs[i + 1][j - 1] > opt_cost:
                    opt_cost = costs[i + 1][j - 1]
                    opt_pos = (i + 1, j - 1)
                costs[i][j] = opt_cost + img[i][j]
                last_opt_node[i][j] = opt_pos

        # find boundary
        max_cost = costs[0][cols - 1]
        max_pos = (0, cols - 1)
        for i in range(rows):
            # print(costs[i][cols - 1])
            if costs[i][cols - 1] > max_cost:
                max_cost = costs[i][cols - 1]
                max_pos = (i, cols - 1)

        li, lj = max_pos
        boundary = [max_pos]

        while last_opt_node[li][lj] != (-1, -1):
            li, lj = last_opt_node[li][lj]
            boundary.append((li, lj))

        return boundary

    def colorBoundary(self, img, boundary, boundary_color):
        for li,lj in boundary:
            img[li][lj] = boundary_color
            if li > 0:
                img[li - 1][lj] = boundary_color
            if li < len(img) - 1:
                img[li + 1][lj] = boundary_color

    def findBrightestLine(self, img, img_gray, color):
        rows, cols = img_gray.shape

        costs = [[0] * cols for i in range(rows)]
        last_opt_node = [[(0, 0)] * cols for i in range(rows)]

        # init
        for i in range(rows):
            costs[i][0] = int(img_gray[i][0])
            last_opt_node[i][0] = (-1, -1)

        # dp
        for j in range(1, cols):
            for i in range(rows):
                opt_cost = costs[i][j - 1]
                opt_pos = (i, j - 1)
                if i > 0 and costs[i - 1][j - 1] > opt_cost:
                    opt_cost = costs[i - 1][j - 1]
                    opt_pos = (i - 1, j - 1)
                if i < rows - 1 and costs[i + 1][j - 1] > opt_cost:
                    opt_cost = costs[i + 1][j - 1]
                    opt_pos = (i + 1, j - 1)
                costs[i][j] = opt_cost + img_gray[i][j]
                last_opt_node[i][j] = opt_pos

        # find boundary
        min_cost = costs[0][cols - 1]
        min_pos = (0, cols - 1)
        for i in range(rows):
            # print(costs[i][cols - 1])
            if costs[i][cols - 1] > min_cost:
                min_cost = costs[i][cols - 1]
                min_pos = (i, cols - 1)

        li, lj = min_pos
        bm_boundary = [min_pos]
        if color == "red":
            boundary_color = (0, 0, 255)
        else:
            boundary_color = (255, 0, 0)

        while last_opt_node[li][lj] != (-1, -1):
            li, lj = last_opt_node[li][lj]
            bm_boundary.append((li, lj))
            img[li][lj] = boundary_color
            # highlight the boundary
            if li > 0:
                img[li - 1][lj] = boundary_color
            if li < rows - 1:
                img[li + 1][lj] = boundary_color

    def getCenterArea(self, img, bm, csj, width):
        rows, cols = img.shape

        mid = int(cols / 2)
        w = int(width / 2)

        ups = dict()
        downs = dict()
        for t in bm:
            if t[1] >= mid - w and t[1] <= mid + w:
                ups[t[1]] = t[0]
        for t in csj:
            if t[1] >= mid - w and t[1] <= mid + w:
                downs[t[1]] = t[0]

        left_boundary = []
        right_boundary = []
        for i in range(ups[mid - w], downs[mid - w]):
            left_boundary.append((i, mid - w))
        for i in range(ups[mid + w], downs[mid + w]):
            right_boundary.append((i, mid + w))

        return ups, downs, left_boundary, right_boundary

    def computeMeanHeight(self, img, ups, downs):
        rows, cols = img.shape

        mid = int(cols / 2)
        width = len(ups)
        w = int(width / 2)

        lens = 0
        for j in range(mid - w, mid + w + 1):
            up = ups[j]
            down = downs[j]
            lens += down - up

        mean_height = lens / width

        # bm may not be horizontal
        y1 = ups[mid - w]
        y2 = ups[mid + w]
        y = abs(y1 - y2) * vertical_scale
        x = len(ups) * horizontal_scale
        rad = math.atan(y / x)

        angle = rad * 180 / math.pi

        res = mean_height * vertical_scale * math.cos(rad)

        return angle, res

    def getBinaryImage(self, img):
        th, img_bi = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        print(th)

        return img_bi

    def contrast(self, img, threshold):
        rows, cols = img.shape

        img_con = np.zeros(img.shape, int)

        for i in range(rows):
            for j in range(cols):
                if img[i][j] < threshold:
                    img_con[i][j] = 0
                else:
                    img_con[i][j] = (img[i][j] - threshold) / (255 - threshold) * 255

        return img_con

    def getNiBlackImage(self, img, ups, downs, n, k):
        rows, cols = img.shape

        niblack = np.zeros(img.shape, int)

        mid = int(cols / 2)
        width = len(ups)
        w = int(width / 2)

        for j in range(mid - w, mid + w + 1):
            for i in range(ups[j], downs[j] + 1):
                mean = 0
                areas = 0
                for x in range(i - n, i + n + 1):
                    for y in range(j - n, j + n + 1):
                        if x > 0 and x < rows and y > 0 and y < cols:
                            mean += img[x][y]
                            areas += 1

                mean = mean / areas

                std = 0
                for x in range(i - n, i + n + 1):
                    for y in range(j - n, j + n + 1):
                        if x > 0 and x < rows and y > 0 and y < cols:
                            std += (img[x][y] - mean) * (img[x][y] - mean)
                std = std / areas
                std = std ** 0.5

                if img[i][j] > mean + k * std:
                    niblack[i][j] = 255
                else:
                    niblack[i][j] = 0

        return niblack

    def computeBlackPercentage(self, niblack, ups, downs):
        rows, cols = niblack.shape

        mid = int(cols / 2)
        width = len(ups)
        w = int(width / 2)

        cnt = 0
        blacks = 0
        for j in range(mid - w, mid + w + 1):
            for i in range(ups[j], downs[j] + 1):
                cnt += 1
                if niblack[i][j] == 0:
                    blacks += 1

        if cnt == 0:
            return -1
        else:
            return blacks / cnt




if __name__ == "__main__":
    """
    b = ChoroidalBoundaryDetection()

    img_path = "/Users/rick/PycharmProjects/ChoroidalBoundary/data/test5.jpg"
    res_path = "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/res5.png"
    niblack_path = "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/niblack5.png"
    binary_threshold = 20
    sba_size = 20
    r1 = 20
    r2 = 25
    center_width = 1500
    niblack_n = 13
    niblack_k = 0.1

    b.detect(img_path, res_path, niblack_path, binary_threshold, sba_size, r1, r2, center_width, niblack_n, niblack_k)
    """

    b = ChoroidalBoundaryDetection()

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    binary_threshold = int(sys.argv[3])
    sba_size = int(sys.argv[4])
    r1 = int(sys.argv[5])
    r2 = int(sys.argv[6])
    center_width = int(sys.argv[7])
    niblack_n = int(sys.argv[8])
    niblack_k = float(sys.argv[9])

    with open(output_dir + "result.txt", "w") as f:
        writer = csv.writer(f)

        for file in os.listdir(input_dir):
            if "jpg" in file:
                print(input_dir + file)

                img_path = input_dir + file
                res_path = output_dir + file.split(".jpg")[0] + "_res.png"
                niblack_path = output_dir + file.split(".jpg")[0] + "_niblack.png"

                angle, mean_height, black_per = b.detect(img_path, res_path, niblack_path, binary_threshold,
                                                         sba_size, r1, r2, center_width, niblack_n, niblack_k)

                line = [img_path, angle, mean_height, black_per]
                writer.writerow(line)


    """
    path = "/Users/rick/PycharmProjects/ChoroidalBoundary/data/test5.jpg"
    img = cv2.imread(path)
    # cut original image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    img = img[430:490, 500:510]

    rows, cols=img.shape

    print(img)

    b = BMDetection()

    b.outputImage(img, "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/test.png")
    """




