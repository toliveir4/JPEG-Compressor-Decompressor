import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np


def encoder():
    img = plt.imread('imagens/peppers.bmp')

    # plt.figure()
    # plt.imshow(img)

    # cm = getColormap()

    # viewColormap(cm, img)
    
    R, G, B = separateRGB(img)

    # viewChanels(R, G, B)

    addPadding(img)

    YCbCr = RGBtoYCbCr(R, G, B)

    # showYCbCr(YCbCr)

    return R, G, B, YCbCr


def getColormap():
    inp = str(input("Introduza colormap: r, g, b "))
    r, g, b = inp.split(",")
    cm = clr.LinearSegmentedColormap.from_list('cm', [(0, 0, 0), (int(r), int(g), int(b))], N=256)
    return cm


def viewColormap(cm, img):
    plt.figure()
    plt.imshow(img, cm)


def viewChanels(R, G, B):
    plt.figure()
    plt.imshow(R, cmr)
    
    plt.figure()
    plt.imshow(G, cmg)
    
    plt.figure()
    plt.imshow(B, cmb)


def separateRGB(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return R, G, B


def addPadding(img):
    h, w, d = np.shape(img)

    if h % 16 != 0:
        add = 16 - (h % 16)
        arr = np.ones(h, dtype=int)
        arr[h-1] = add + 1
        img = np.repeat(img, arr, axis=0)

    if w % 16 != 0:
        add = 16 - (w % 16)
        arr = np.ones(w, dtype=int)
        arr[w-1] = add + 1
        img = np.repeat(img, arr, axis=1)


def RGBtoYCbCr(R, G, B):
    matrix = np.array([[0.299, 0.587, 0.114], 
                    [-0.168736, -0.331264, 0.5], 
                    [0.5, -0.418688, -0.081312]])

    RGB = joinRGB(R, G, B)

    YCbCr = RGB.dot(matrix)
    YCbCr[:, :, [1, 2]] += 128

    return YCbCr


def decoder(R, G, B, YCbCr):
    RGBAfter = YCbCrtoRGB(YCbCr)
    RGBBefore = joinRGB(R, G, B)
    
    # plt.figure()
    # plt.imshow(RGBAfter)

    # comp = RGBAfter == RGBBefore
    # res = comp.all()
    # print(res)

def joinRGB(R, G, B):
    RGB = np.dstack((R, G, B))

    return RGB


def YCbCrtoRGB(YCbCr):
    matrix = np.array([[0.299, 0.587, 0.114], 
                    [-0.168736, -0.331264, 0.5], 
                    [0.5, -0.418688, -0.081312]])

    inverted = np.linalg.inv(matrix)
    YCbCr[:, :, [1, 2]] -= 128
    RGB = YCbCr.dot(inverted)
    RGB = RGB.round()
    RGB[RGB > 255] = 255
    RGB[RGB < 0] = 0
    RGB = RGB.astype(np.uint8)

    return RGB


def showYCbCr(YCbCr):
    Y = YCbCr[:, :, 0]
    Cb = YCbCr[:, :, 1]
    Cr = YCbCr[:, :, 2]

    plt.figure()
    plt.imshow(Y, cmGray)
    plt.figure()
    plt.imshow(Cb, cmGray)
    plt.figure()
    plt.imshow(Cr, cmGray)


def main():
    plt.close('all')

    R, G, B, YCbCr = encoder()
    decoder(R, G, B, YCbCr)

    plt.show()


if __name__ == '__main__':
    cmr = clr.LinearSegmentedColormap.from_list('cmr', [(0, 0, 0), (1, 0, 0)], N=256)
    cmg = clr.LinearSegmentedColormap.from_list('cmg', [(0, 0, 0), (0, 1, 0)], N=256)
    cmb = clr.LinearSegmentedColormap.from_list('cmb', [(0, 0, 0), (0, 0, 1)], N=256)
    cmGray = clr.LinearSegmentedColormap.from_list('cmGray', [(0, 0, 0), (1, 1, 1)], N=256)
    main()
