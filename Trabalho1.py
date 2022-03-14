import matplotlib.colors as cb_linesr
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct


def encoder():  # 2
    img_name = "barn_mountains"

    global dSample, opt
    dSample = 420
    opt = 8

    img = plt.imread(f'imagens/{img_name}.bmp')  # 3.1

    '''plt.figure()
    plt.imshow(img)
    plt.axis('off')'''

    # cm = getColormap() # 3.2

    # viewColormap(cm, img) # 3.3

    img = addPadding(img)  # 4.1

    R, G, B = separateRGB(img)  # 3.4

    # viewChanels(R, G, B) # 3.5

    YCbCr = RGBtoYCbCr(R, G, B)  # 5

    # showYCbCr(YCbCr) # 5

    YD, CbD, CrD = downSample(YCbCr, dSample)  # 6

    # showDownSample(YD, CbD, CrD) # 6

    Y_dct, Cb_dct, Cr_dct = calcDCT_8x8_64x64(YD, CbD, CrD)  # 7
    # showDCT(Y_dct, Cb_dct, Cr_dct) # 7.1.2

    Y_Q, Cb_Q, Cr_Q = quantization(Y_dct, Cb_dct, Cr_dct) # 8

    Y_dcpm, Cb_dcpm, Cr_dcpm = DCPM(Y_Q, Cb_Q, Cr_Q) # 9

    return Y_dcpm, Cb_dcpm, Cr_dcpm


def getColormap():  # 3.2
    inp = str(input("Introduza colormap (r, g, b): "))
    r, g, b = inp.split(",")
    cm = cb_linesr.LinearSegmentedColormap.from_list(
        'cm', [(0, 0, 0), (int(r), int(g), int(b))], N=256)
    return cm


def viewColormap(cm, img):  # 3.3
    plt.figure()
    plt.imshow(img, cm)


def viewChanels(R, G, B):  # 3.5
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(131)
    plt.title("Red")
    plt.imshow(R, cmr)
    fig.add_subplot(132)
    plt.title("Green")
    plt.imshow(G, cmg)
    fig.add_subplot(133)
    plt.title("Blue")
    plt.imshow(B, cmb)


def separateRGB(img):  # 3.4
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return R, G, B


def addPadding(img):  # 4.1
    global h, w
    h, w, _ = np.shape(img)
    if opt == 64:
        pad = 128
    else:
        pad = 16

    if h % pad != 0:
        add = pad - (h % pad)
        arr = np.ones(h, dtype=int)
        arr[h-1] = add + 1
        img = np.repeat(img, arr, axis=0)

    if w % pad != 0:
        add = pad - (w % pad)
        arr = np.ones(w, dtype=int)
        arr[w-1] = add + 1
        img = np.repeat(img, arr, axis=1)
    return img


def RGBtoYCbCr(R, G, B):  # 5
    matrix = np.array([[0.299, 0.587, 0.114],
                       [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]])

    RGB = joinRGB(R, G, B)

    YCbCr = RGB.dot(matrix.T)
    YCbCr[:, :, [1, 2]] += 128

    return YCbCr


def separateYCbCr(YCbCr):
    Y = YCbCr[:, :, 0]
    Cb = YCbCr[:, :, 1]
    Cr = YCbCr[:, :, 2]

    return Y, Cb, Cr


def showYCbCr(YCbCr):  # 5
    Y, Cb, Cr = separateYCbCr(YCbCr)

    fig = plt.figure(figsize=(10, 5))
    plt.title("YCbCr")
    plt.axis("off")
    fig.add_subplot(131)
    plt.title("Y")
    plt.imshow(Y, cmGray)
    fig.add_subplot(132)
    plt.title("Cb")
    plt.imshow(Cb, cmGray)
    fig.add_subplot(133)
    plt.title("Cr")
    plt.imshow(Cr, cmGray)


def downSample(YCbCr, dSample):  # 6
    Y, Cb, Cr = separateYCbCr(YCbCr)

    Cb = Cb[:, ::2]
    Cr = Cr[:, ::2]

    if(dSample == 420):
        Cb = Cb[::2, :]
        Cr = Cr[::2, :]

    return Y, Cb, Cr


def showDownSample(YD, CbD, CrD):  # 6
    fig = plt.figure(figsize=(10, 5))
    plt.title("YCbCr")
    plt.axis("off")
    fig.add_subplot(131)
    plt.title("Y")
    plt.imshow(YD, cmGray)
    fig.add_subplot(132)
    plt.title("Cb Downsampled")
    plt.imshow(CbD, cmGray)
    fig.add_subplot(133)
    plt.title("Cr Downsampled")
    plt.imshow(CrD, cmGray)


def calcDCT_8x8_64x64(YD, CbD, CrD):  # 7
    if opt == 8 or opt == 64:
        Y_lines, Y_cols = np.shape(YD)
        Cb_lines, Cb_cols = np.shape(CbD)

        Y_dct = YD
        Cb_dct = CbD
        Cr_dct = CrD

        for i in range(0, Y_lines, opt):
            for j in range(0, Y_cols, opt):
                Y_dct[i:i+opt, j:j+opt] = dct(dct(Y_dct[i:i + opt,j:j + opt], norm="ortho").T, norm="ortho").T

        for i in range(0, Cb_lines, opt):
            for j in range(0, Cb_cols, opt):
                Cb_dct[i:i + opt, j:j + opt] = dct(dct(Cb_dct[i:i + opt, j:j + opt], norm="ortho").T, norm="ortho").T
                Cr_dct[i:i + opt, j:j + opt] = dct(dct(Cr_dct[i:i + opt, j:j + opt], norm="ortho").T, norm="ortho").T
    else:
        Y_dct = dct(dct(YD, norm='ortho').T, norm='ortho').T
        Cb_dct = dct(dct(CbD, norm='ortho').T, norm='ortho').T
        Cr_dct = dct(dct(CrD, norm='ortho').T, norm='ortho').T
    
    return Y_dct, Cb_dct, Cr_dct


def showDCT(Y_dct, Cb_dct, Cr_dct):  # 7.1.2
    fig = plt.figure(figsize=(12, 5))
    fig.add_subplot(131)
    plt.title("Y_DCT")
    plt.imshow(np.log(abs(Y_dct) + 0.0001), cmGray)
    fig.add_subplot(132)
    plt.title("Cb_DCT")
    plt.imshow(np.log(abs(Cb_dct) + 0.0001), cmGray)
    fig.add_subplot(133)
    plt.title("Cr_DCT")
    plt.imshow(np.log(abs(Cr_dct) + 0.0001), cmGray)


def getQuantizationMatrix(requiredQuality=50):      # 8
    Q_Y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

    Q_CbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])

    if requiredQuality == 50:
        return Q_Y, Q_CbCr
    elif requiredQuality > 50:
        Q_Y = (Q_Y * ((100 - requiredQuality) / 50)).round()
        Q_Y[Q_Y > 255] = 255
        Q_Y = Q_Y.astype(np.uint8)

        Q_CbCr = (Q_CbCr * ((100 - requiredQuality) / 50)).round()
        Q_CbCr[Q_CbCr > 255] = 255
        Q_CbCr = Q_CbCr.astype(np.uint8)
        return Q_Y, Q_CbCr
    else:
        Q_Y = (Q_Y * (50 / requiredQuality)).round()
        Q_Y[Q_Y > 255] = 255
        Q_Y = Q_Y.astype(np.uint8)

        Q_CbCr = (Q_CbCr * (50 / requiredQuality)).round()
        Q_CbCr[Q_CbCr > 255] = 255
        Q_CbCr = Q_CbCr.astype(np.uint8)
        return Q_Y, Q_CbCr


def quantization(Y_dct, Cb_dct, Cr_dct):    # 8
    Q_Y, Q_CbCr = getQuantizationMatrix(requiredQuality=75)

    Y_lines, Y_cols = np.shape(Y_dct)
    Cb_lines, Cb_cols = np.shape(Cb_dct)

    YQ , CbQ, CrQ = Y_dct, Cb_dct, Cr_dct

    for i in range(0, Y_lines, opt):
        for j in range(0, Y_cols, opt):
            YQ[i:i + opt, j:j + opt] = np.round(YQ[i:i + opt, j:j + opt] / Q_Y)
            
    for i in range(0, Cb_lines, opt):
        for j in range(0, Cb_cols, opt):
            CbQ[i:i + opt, j:j + opt] = np.round(CbQ[i:i + opt, j:j + opt] / Q_CbCr)
            CrQ[i:i + opt, j:j + opt] = np.round(CrQ[i:i + opt, j:j + opt] / Q_CbCr)

    return YQ, CbQ, CrQ


def DCPM(YQ, CbQ, CrQ): # 9
    Y_lines, Y_cols = np.shape(YQ)
    C_lines, C_cols = np.shape(CbQ)

    dcY0 = YQ[0, 0]
    dcCb0 = CbQ[0, 0]
    dcCr0 = CrQ[0, 0]

    for i in range(8, Y_lines, 8):
        for j in range(8, Y_cols, 8):
            dcY = YQ[i, j]
            diffY = dcY - dcY0
            YQ[i, j] = diffY
            dcY0 = dcY
            if i < C_lines and j < C_cols:
                dcCb = CbQ[i, j]
                dcCr = CrQ[i, j]

                diffCb = dcCb - dcCb0
                diffCr = dcCr - dcCr0

                CbQ[i, j] = diffCb
                CrQ[i, j] = diffCr

                dcCb0 = dcCb
                dcCr0 = dcCr

    return YQ, CbQ, CrQ


def decoder(Y_dcpm, Cb_dcpm, Cr_dcpm):  # 2
    Y_Q, Cb_Q, Cr_Q = IDCPM(Y_dcpm, Cb_dcpm, Cr_dcpm)

    Y_dct, Cb_dct, Cr_dct = deQuantization(Y_Q, Cb_Q, Cr_Q) # 8

    Y_enc, Cb_enc, Cr_enc = calcIDCT_8x8_64x64(Y_dct, Cb_dct, Cr_dct)  # 7

    YCbCrU = upSample(Y_enc, Cb_enc, Cr_enc, dSample)  # 6

    # showYCbCr(YCbCrU) # 6

    RGBAfter = YCbCrtoRGB(YCbCrU)  # 5

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

def IDCPM(Y_dcpm, Cb_dcpm, Cr_dcpm): # 9
    Y_lines, Y_cols = np.shape(Y_dcpm)
    C_lines, C_cols = np.shape(Cb_dcpm)

    YQ , CbQ, CrQ = Y_dcpm, Cb_dcpm, Cr_dcpm

    dcY0 = YQ[0, 0]
    dcCb0 = CbQ[0, 0]
    dcCr0 = CrQ[0, 0]

    for i in range(8, Y_lines, 8):
        for j in range(8, Y_cols, 8):
            dcY = YQ[i, j]
            diffY = dcY + dcY0
            YQ[i, j] = diffY
            dcY0 = dcY

            if i < C_lines and j < C_cols:
                dcCb = CbQ[i, j]
                dcCr = CrQ[i, j]

                diffCb = dcCb + dcCb0
                diffCr = dcCr + dcCr0

                CbQ[i, j] = diffCb
                CrQ[i, j] = diffCr

                dcCb0 = dcCb
                dcCr0 = dcCr

    return YQ, CbQ, CrQ


def deQuantization(Y_Q, Cb_Q, Cr_Q):    # 8
    Q_Y, Q_CbCr = getQuantizationMatrix(requiredQuality=75)

    Y_lines, Y_cols = np.shape(Y_Q)
    Cb_lines, Cb_cols = np.shape(Cb_Q)

    Y_dct, Cb_dct, Cr_dct = Y_Q, Cb_Q, Cr_Q

    for i in range(0, Y_lines, opt):
        for j in range(0, Y_cols, opt):
            Y_dct[i:i + opt, j:j + opt] = Y_dct[i:i + opt, j:j + opt] * Q_Y
                
    for i in range(0, Cb_lines, opt):
        for j in range(0, Cb_cols, opt):
            Cb_dct[i:i + opt, j:j + opt] = Cb_dct[i:i + opt, j:j + opt] * Q_CbCr
            Cr_dct[i:i + opt, j:j + opt] = Cr_dct[i:i + opt, j:j + opt] * Q_CbCr
    
    return Y_dct, Cb_dct, Cr_dct


def calcIDCT_8x8_64x64(Y_dct, Cb_dct, Cr_dct):  # 7
    if opt == 8 or opt == 64:
        Y_lines, Y_cols = np.shape(Y_dct)
        Cb_lines, Cb_cols = np.shape(Cb_dct)

        Y = Y_dct
        Cb = Cb_dct
        Cr = Cr_dct

        for i in range(0, Y_lines, opt):
            for j in range(0, Y_cols, opt):
                Y[i:i+opt, j:j+opt] = idct(idct(Y[i:i + opt,j:j + opt], norm="ortho").T, norm="ortho").T

        for i in range(0, Cb_lines, opt):
            for j in range(0, Cb_cols, opt):
                Cb[i:i + opt, j:j + opt] = idct(idct(Cb[i:i + opt, j:j + opt], norm="ortho").T, norm="ortho").T
                Cr[i:i + opt, j:j + opt] = idct(idct(Cr[i:i + opt, j:j + opt], norm="ortho").T, norm="ortho").T

    else:
        Y = idct(idct(Y_dct, norm='ortho').T, norm='ortho').T
        Cb = idct(idct(Cb_dct, norm='ortho').T, norm='ortho').T
        Cr = idct(idct(Cr_dct, norm='ortho').T, norm='ortho').T

    return Y, Cb, Cr


def upSample(YD, CbD, CrD, dSample):  # 6
    CbU = np.repeat(CbD, 2, axis=1)
    CrU = np.repeat(CrD, 2, axis=1)

    if np.shape(YD)[0] % 2 != 0:
        CbU = np.delete(CbU, -1, 0)
        CrU = np.delete(CrU, -1, 0)

    if dSample == 420:
        CbU = np.repeat(CbU, 2, axis=0)
        CrU = np.repeat(CrU, 2, axis=0)

        if np.shape(YD)[1] % 2 != 0:
            CbU = np.delete(CbU, -1, 1)
            CrU = np.delete(CrU, -1, 1)

    YCbCrU = np.dstack((YD, CbU, CrU))

    return YCbCrU


def YCbCrtoRGB(YCbCr):  # 5
    matrix = np.array([[0.299, 0.587, 0.114],
                       [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]])

    inverted = np.linalg.inv(matrix)
    YCbCr[:, :, [1, 2]] -= 128
    RGB = YCbCr.dot(inverted.T)
    RGB = RGB.round()
    RGB[RGB > 255] = 255
    RGB[RGB < 0] = 0
    RGB = RGB.astype(np.uint8)

    return RGB


def unpadding(RGBAfter):  # 4
    R, G, B = separateRGB(RGBAfter)

    R = R[:h, :w]
    G = G[:h, :w]
    B = B[:h, :w]

    return joinRGB(R, G, B)


def joinRGB(R, G, B):  # 3.4
    RGB = np.dstack((R, G, B))

    return RGB


def main():
    plt.close('all')

    Y_dcpm, Cb_dcpm, Cr_dcpm = encoder()
    decoder(Y_dcpm, Cb_dcpm, Cr_dcpm)

    plt.show()


if __name__ == '__main__':
    cmr = cb_linesr.LinearSegmentedColormap.from_list(
        'cmr', [(0, 0, 0), (1, 0, 0)], N=256)
    cmg = cb_linesr.LinearSegmentedColormap.from_list(
        'cmg', [(0, 0, 0), (0, 1, 0)], N=256)
    cmb = cb_linesr.LinearSegmentedColormap.from_list(
        'cmb', [(0, 0, 0), (0, 0, 1)], N=256)
    cmGray = cb_linesr.LinearSegmentedColormap.from_list(
        'cmGray', [(0, 0, 0), (1, 1, 1)], N=256)

    main()
