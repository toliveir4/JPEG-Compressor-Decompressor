import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np


def encoder(): # 2
    img = plt.imread('imagens/peppers.bmp') # 3.1

    # plt.figure()
    # plt.imshow(img)

    # cm = getColormap() # 3.2

    # viewColormap(cm, img) # 3.3
    
    R, G, B = separateRGB(img) # 3.4

    # viewChanels(R, G, B) # 3.5

    addPadding(img) # 4.1

    YCbCr = RGBtoYCbCr(R, G, B) # 5

    # showYCbCr(YCbCr) # 5

    # YD, CbD, CrD = downSample422(YCbCr) # 6

    YD, CbD, CrD = downSample420(YCbCr) # 6

    return R, G, B, YD, CbD, CrD


def getColormap(): # 3.2
    inp = str(input("Introduza colormap: r, g, b "))
    r, g, b = inp.split(",")
    cm = clr.LinearSegmentedColormap.from_list('cm', [(0, 0, 0), (int(r), int(g), int(b))], N=256)
    return cm


def viewColormap(cm, img): # 3.3
    plt.figure()
    plt.imshow(img, cm)


def viewChanels(R, G, B): # 3.5
    plt.figure()
    plt.imshow(R, cmr)
    
    plt.figure()
    plt.imshow(G, cmg)
    
    plt.figure()
    plt.imshow(B, cmb)


def separateRGB(img): # 3.4
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return R, G, B


def addPadding(img): # 4.1
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


def RGBtoYCbCr(R, G, B): # 5
    matrix = np.array([[0.299, 0.587, 0.114], 
                    [-0.168736, -0.331264, 0.5], 
                    [0.5, -0.418688, -0.081312]])

    RGB = joinRGB(R, G, B)

    YCbCr = RGB.dot(matrix)
    YCbCr[:, :, [1, 2]] += 128

    return YCbCr


def downSample422(YCbCr): # 6
    Y = YCbCr[:, :, 0]
    Cb = YCbCr[:, :, 1]
    Cr = YCbCr[:, :, 2]

    Cb = Cb[:, ::2]
    Cr = Cr[:, ::2]

    return Y, Cb, Cr


def downSample420(YCbCr): # 6
    Y = YCbCr[:, :, 0]
    Cb = YCbCr[:, :, 1]
    Cr = YCbCr[:, :, 2]

    Cb = Cb[:, ::2]
    Cb = Cb[::2, :]
    Cr = Cr[:, ::2]
    Cr = Cr[::2, :]

    return Y, Cb, Cr


def decoder(R, G, B, YD, CbD, CrD): # 2
    # YCbCrU = upSample422(YD, CbD, CrD) # 6

    # YCbCrU = upSample420(YD, CbD, CrD) # 6
    
    # showYCbCr(YCbCrU) # 6

    # RGBAfter = YCbCrtoRGB(YCbCrU) # 5
    # RGBBefore = joinRGB(R, G, B) # 3.4
    
    '''plt.figure()
    plt.imshow(RGBAfter)

    comp = RGBAfter == RGBBefore
    res = comp.all()
    print(res)'''

def joinRGB(R, G, B): # 3.4
    RGB = np.dstack((R, G, B))

    return RGB


def YCbCrtoRGB(YCbCr): # 5
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


def upSample422(YD, CbD, CrD): # 6
    CbU = np.repeat(CbD, 2, axis=1)
    
    CrU = np.repeat(CrD, 2, axis=1)

    YCbCrU = np.dstack((YD, CbU, CrU))

    return YCbCrU


def upSample420(YD, CbD, CrD): # 6
    CbU = np.repeat(CbD, 2, axis=1)
    CbU = np.repeat(CbU, 2, axis=0)
    CrU = np.repeat(CrD, 2, axis=1)
    CrU = np.repeat(CrU, 2, axis=0)

    YCbCrU = np.dstack((YD, CbU, CrU))

    return YCbCrU


def showYCbCr(YCbCr): # 5
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

    R, G, B, YD, CbD, CrD = encoder()
    decoder(R, G, B, YD, CbD, CrD)

    plt.show()


if __name__ == '__main__':
    cmr = clr.LinearSegmentedColormap.from_list('cmr', [(0, 0, 0), (1, 0, 0)], N=256)
    cmg = clr.LinearSegmentedColormap.from_list('cmg', [(0, 0, 0), (0, 1, 0)], N=256)
    cmb = clr.LinearSegmentedColormap.from_list('cmb', [(0, 0, 0), (0, 0, 1)], N=256)
    cmGray = clr.LinearSegmentedColormap.from_list('cmGray', [(0, 0, 0), (1, 1, 1)], N=256)
    main()
