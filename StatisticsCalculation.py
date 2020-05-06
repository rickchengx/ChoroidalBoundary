import cv2
import numpy as np
import os
import math
import sys
import csv

horizontal_scale = 1500 / 120
vertical_scale = 200 / 51

class StatisticsCalculation:
    def checkRed(self, hsvcolor):
        (h, s, v) = hsvcolor

        if ((h >= 0 and h <= 10) or (h >= 156 and h <= 180)) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
            return True
        else:
            return False

    def colorBoundary(self, img, boundary, boundary_color):
        for li,lj in boundary:
            img[li][lj] = boundary_color
            if li > 0:
                img[li - 1][lj] = boundary_color
            if li < len(img) - 1:
                img[li + 1][lj] = boundary_color

    def detect(self, img_path, res_path, niblack_path, center_width, niblack_n, niblack_k):
        img = cv2.imread(img_path)
        # cut original image
        img = img[10:490, 582:1182]

        width = center_width / horizontal_scale
        ups, downs, left_boundary, right_boundary = self.getCenterArea(img, width)
        self.colorBoundary(img, left_boundary, (255, 0, 0))
        self.colorBoundary(img, right_boundary, (255, 0, 0))

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        angle, mean_height = self.computeMeanHeight(ups, downs)
        print(angle, mean_height)

        niblack = self.getNiBlackImage(img_gray, ups, downs, niblack_n, niblack_k)
        black_per = self.computeBlackPercentage(niblack, ups, downs)
        self.outputImage(niblack, niblack_path)
        print(black_per)

        self.outputImage(img, res_path)

        return angle, mean_height, black_per

    def getCenterArea(self, img, width):
        rows, cols, cc = img.shape

        mid = int(cols / 2)
        w = int(width / 2)

        ups = dict()
        downs = dict()

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for j in range(mid - w, mid + w + 1):
            up = -1
            down = -1
            for i in range(rows):
                hsvcolor = img_hsv[i][j]
                if self.checkRed(hsvcolor):

                    if up == -1:
                        up = i
                    else:
                        if i - up > 5:
                            down = i
            if up != -1 and down != -1:
                ups[j] = up
                downs[j] = down

        left_boundary = []
        right_boundary = []

        keys = list(ups.keys())
        keys.sort()

        for i in range(ups[keys[0]], downs[keys[0]]):
            left_boundary.append((i, keys[0]))
        for i in range(ups[keys[len(keys) - 1]], downs[keys[len(keys) - 1]]):
            right_boundary.append((i, keys[len(keys) - 1]))

        return ups, downs, left_boundary, right_boundary

    def computeMeanHeight(self, ups, downs):
        width = len(ups)

        lens = 0
        keys = list(ups.keys())
        keys.sort()

        for t in range(len(keys)):
            j = keys[t]
            up = ups[j]
            down = downs[j]
            lens += down - up

        mean_height = lens / width

        # bm may not be horizontal
        y1 = ups[keys[0]]
        y2 = ups[keys[len(keys) - 1]]
        y = abs(y1 - y2) * vertical_scale
        x = len(ups) * horizontal_scale
        rad = math.atan(y / x)

        angle = rad * 180 / math.pi

        res = mean_height * vertical_scale * math.cos(rad)

        return angle, res

    def getNiBlackImage(self, img, ups, downs, n, k):
        rows, cols = img.shape

        niblack = np.zeros(img.shape, int)

        keys = list(ups.keys())
        keys.sort()

        for t in range(len(keys)):
            j = keys[t]
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
        keys = list(ups.keys())
        keys.sort()

        cnt = 0
        blacks = 0
        for t in range(len(keys)):
            j = keys[t]
            for i in range(ups[j], downs[j] + 1):
                cnt += 1
                if niblack[i][j] == 0:
                    blacks += 1

        if cnt == 0:
            return -1
        else:
            return blacks / cnt

    def rectangleTest(self, img_path, out_path):
        img = cv2.imread(img_path)
        # cut original image
        img = img[10:490, 582:1182]

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rows, cols, cc = img.shape

        rowmin = rows
        rowmax = 0
        colmin = cols
        colmax = 0
        for i in range(rows):
            for j in range(cols):
                hsvcolor = img_hsv[i][j]
                if self.checkRed(hsvcolor):
                    if i < rowmin:
                        rowmin = i
                    if i > rowmax:
                        rowmax = i
                    if j < colmin:
                        colmin = j
                    if j > colmax:
                        colmax = j

        img_cut = img[rowmin:rowmax, colmin:colmax]

        self.outputImage(img_cut, out_path)

    def outputImage(self, img, path):
        cv2.imwrite(path, img,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 3])



if __name__ == "__main__":
    """
    img_path = "/Users/rick/PycharmProjects/ChoroidalBoundary/data/OCT_2/1.png"
    res_path = "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/res.png"
    niblack_path = "/Users/rick/PycharmProjects/ChoroidalBoundary/data/res/niblack.png"
    s = StatisticsCalculation()
    s.detect(img_path, res_path,niblack_path, 1500, 13, 0.1)
    """

    s = StatisticsCalculation()

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    center_width = int(sys.argv[3])
    niblack_n = int(sys.argv[4])
    niblack_k = float(sys.argv[5])

    with open(output_dir + "result.txt", "w") as f:
        writer = csv.writer(f)

        for file in os.listdir(input_dir):
            if "jpg" in file:
                print(input_dir + file)

                img_path = input_dir + file
                res_path = output_dir + file.split(".jpg")[0] + "_res.png"
                niblack_path = output_dir + file.split(".jpg")[0] + "_niblack.png"

                angle, mean_height, black_per = s.detect(img_path, res_path, niblack_path, center_width, niblack_n,
                                                         niblack_k)

                line = [img_path, angle, mean_height, black_per]
                writer.writerow(line)
            elif "png" in file:
                print(input_dir + file)

                img_path = input_dir + file
                res_path = output_dir + file.split(".png")[0] + "_res.png"
                niblack_path = output_dir + file.split(".png")[0] + "_niblack.png"

                angle, mean_height, black_per = s.detect(img_path, res_path, niblack_path, center_width, niblack_n,
                                                         niblack_k)

                line = [img_path, angle, mean_height, black_per]
                writer.writerow(line)


