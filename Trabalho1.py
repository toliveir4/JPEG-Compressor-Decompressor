import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np


def encoder():
    img = plt.imread('imagens/logo.bmp')

    #cm = getColormap()

    #viewColormap(cm, img)
    
    R, G, B = separateRGB(img)
    
    RGB = joinRGB(R, G, B)

    #viewChanels(R, G, B)

    addPadding(img)


def getColormap():
    inp = str(input("Introduza colormap: r, g, b "))
    r, g, b = inp.split(",")
    cm = clr.LinearSegmentedColormap.from_list('cm', [(0, 0, 0), (int(r), int(g), int(b))], N=256)
    return cm


def viewColormap(cm, img):
    plt.figure()
    plt.imshow(img, cm)


def viewChanels(R, G, B):
    cmr = clr.LinearSegmentedColormap.from_list('cmr', [(0, 0, 0), (1, 0, 0)], N=256)
    cmg = clr.LinearSegmentedColormap.from_list('cmg', [(0, 0, 0), (0, 1, 0)], N=256)
    cmb = clr.LinearSegmentedColormap.from_list('cmb', [(0, 0, 0), (0, 0, 1)], N=256)

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


def joinRGB(R, G, B):
    RGB = np.dstack((R, G, B))

    return RGB


def addPadding(img):
    h, w, d = np.shape(img)

    print("%d, %d" % (h, w))
    plt.figure()
    plt.imshow(img)

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

    print(np.shape(img))
    plt.figure()
    plt.imshow(img)



def decoder():
    return


def main():
    plt.close('all')

    encoder()
    
    plt.show()


if __name__ == '__main__':
    main()
