import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct


def encoder(): # 2
    img_name = str(input("Image name: "))
    img = plt.imread(f'imagens/{img_name}.bmp') # 3.1

    '''plt.figure()
    plt.imshow(img)'''

    # cm = getColormap() # 3.2

    # viewColormap(cm, img) # 3.3

    img = addPadding(img) # 4.1

    R, G, B = separateRGB(img) # 3.4

    #viewChanels(R, G, B) # 3.5
    
    YCbCr = RGBtoYCbCr(R, G, B) # 5

    # showYCbCr(YCbCr) # 5

    # YD, CbD, CrD = downSample422(YCbCr) # 6
    YD, CbD, CrD = downSample420(YCbCr) # 6

    """plt.figure()
    plt.imshow(YD, cmGray)
    
    plt.figure()
    plt.imshow(CbD, cmGray)
    
    plt.figure()
    plt.imshow(CrD, cmGray)"""

    Y_dct, Cb_dct, Cr_dct = calcDCT(YD, CbD, CrD) # 7.1

    
    """#7.1.2
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(131)
    plt.imshow(np.log(abs(Y_dct) + 0.0001))
    fig.add_subplot(132)
    plt.imshow(np.log(abs(Cb_dct) + 0.0001))
    fig.add_subplot(133)
    plt.imshow(np.log(abs(Cr_dct) + 0.0001))
    plt.colorbar()"""
    

    return Y_dct, Cb_dct, Cr_dct


def getColormap(): # 3.2
    inp = str(input("Introduza colormap (r, g, b): "))
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
    global h, w
    h, w, _ = np.shape(img)

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
    return img


def RGBtoYCbCr(R, G, B): # 5
    matrix = np.array([[0.299, 0.587, 0.114], 
                    [-0.168736, -0.331264, 0.5], 
                    [0.5, -0.418688, -0.081312]])

    RGB = joinRGB(R, G, B)

    YCbCr = RGB.dot(matrix)
    YCbCr[:, :, [1, 2]] += 128

    return YCbCr


def separateYCbCr(YCbCr):
    Y = YCbCr[:, :, 0]
    Cb = YCbCr[:, :, 1]
    Cr = YCbCr[:, :, 2]
    
    return Y, Cb, Cr


def downSample422(YCbCr): # 6
    Y, Cb, Cr = separateYCbCr(YCbCr)

    Cb = Cb[:, ::2]
    Cr = Cr[:, ::2]

    return Y, Cb, Cr


def downSample420(YCbCr): # 6
    Y, Cb, Cr = separateYCbCr(YCbCr)

    Cb = Cb[:, ::2]
    Cb = Cb[::2, :]
    Cr = Cr[:, ::2]
    Cr = Cr[::2, :]

    return Y, Cb, Cr


def calcDCT(YD, CbD, CrD): # 7.1
    Y_dct = dct(dct(YD, norm='ortho').T, norm='ortho').T
    Cb_dct = dct(dct(CbD, norm='ortho').T, norm='ortho').T
    Cr_dct = dct(dct(CrD, norm='ortho').T, norm='ortho').T

    return Y_dct, Cb_dct, Cr_dct


def decoder(Y_dct, Cb_dct, Cr_dct): # 2
    Y_enc, Cb_enc, Cr_enc = calcIDCT(Y_dct, Cb_dct, Cr_dct) # 7.1

    # YCbCrU = upSample422(Y_enc, Cb_enc, Cr_enc) # 6

    YCbCrU = upSample420(Y_enc, Cb_enc, Cr_enc) # 6

    # showYCbCr(YCbCrU) # 6

    RGBAfter = YCbCrtoRGB(YCbCrU) # 5
    
    # RGBBefore = joinRGB(R, G, B) # 3.4
    
    '''plt.figure()
    plt.imshow(RGBAfter)'''

    '''comp = RGBAfter == RGBBefore
    res = comp.all()
    print(res)'''
    
    RGBAfter = unpadding(RGBAfter)  # 4
    
    plt.figure()
    plt.title("Imagem reconstruida")
    plt.imshow(RGBAfter)
    plt.axis('off')


def joinRGB(R, G, B): # 3.4
    RGB = np.dstack((R, G, B))

    return RGB


def unpadding(RGBAfter): # 4
    R, G, B = separateRGB(RGBAfter)
    
    R = R[:h, :w]
    G = G[:h, :w]
    B = B[:h, :w]

    return joinRGB(R, G, B)


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

    if np.shape(YD)[0] % 2 != 0:
        CbU = np.delete(CbU, -1, 0)
        CrU = np.delete(CrU, -1, 0)

    YCbCrU = np.dstack((YD, CbU, CrU))

    return YCbCrU


def upSample420(YD, CbD, CrD): # 6
    CbU = np.repeat(CbD, 2, axis=1)
    CbU = np.repeat(CbU, 2, axis=0)
    CrU = np.repeat(CrD, 2, axis=1)
    CrU = np.repeat(CrU, 2, axis=0)

    if np.shape(YD)[0] % 2 != 0:
        CbU = np.delete(CbU, -1, 0)
        CrU = np.delete(CrU, -1, 0)

    if np.shape(YD)[1] % 2 != 0:
        CbU = np.delete(CbU, -1, 1)
        CrU = np.delete(CrU, -1, 1)

    YCbCrU = np.dstack((YD, CbU, CrU))

    return YCbCrU


def showYCbCr(YCbCr): # 5
    Y, Cb, Cr = separateYCbCr(YCbCr)

    plt.figure()
    plt.imshow(Y, cmGray)
    plt.figure()
    plt.imshow(Cb, cmGray)
    plt.figure()
    plt.imshow(Cr, cmGray)


def calcIDCT(Y_dct, Cb_dct, Cr_dct):
    Y = idct(idct(Y_dct, norm='ortho').T, norm='ortho').T
    Cb = idct(idct(Cb_dct, norm='ortho').T, norm='ortho').T
    Cr = idct(idct(Cr_dct, norm='ortho').T, norm='ortho').T
    
    return Y, Cb, Cr


def main():
    plt.close('all')

    Y_dct, Cb_dct, Cr_dct = encoder()
    decoder(Y_dct, Cb_dct, Cr_dct)

    plt.show()


if __name__ == '__main__':
    cmr = clr.LinearSegmentedColormap.from_list('cmr', [(0, 0, 0), (1, 0, 0)], N=256)
    cmg = clr.LinearSegmentedColormap.from_list('cmg', [(0, 0, 0), (0, 1, 0)], N=256)
    cmb = clr.LinearSegmentedColormap.from_list('cmb', [(0, 0, 0), (0, 0, 1)], N=256)
    cmGray = clr.LinearSegmentedColormap.from_list('cmGray', [(0, 0, 0), (1, 1, 1)], N=256)
    main()
