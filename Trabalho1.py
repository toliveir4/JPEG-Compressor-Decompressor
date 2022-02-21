import matplotlib.colors as clr
import matplotlib.pyplot as plt


def main():
    plt.close('all')

    img = plt.imread('imagens/peppers.bmp')
    
    plt.figure()
    plt.imshow(img)
    
    print(img.shape)

    # R = img[:, :, 0]
    # G = img[:, :, 1]
    B = img[:, :, 2]
    print(B[0, 0])
    print(B.dtype)

    plt.figure()
    plt.imshow(B)
    

    cmGray = clr.LinearSegmentedColormap.from_list('mygray', [(0, 0, 0), (1, 1, 1)], N=256)
    cmRed = clr.LinearSegmentedColormap.from_list('myred', [(0, 0, 0), (1, 0, 0)], N=256)
    cmBlue = clr.LinearSegmentedColormap.from_list('myblue', [(0, 0, 0), (0, 0, 1)], N=256)
    plt.figure()
    plt.imshow(B, cmGray)
    plt.show()

    '''
    encoder(img, ...)
    imgRec = decoder(...)
    '''


if __name__ == '__main__':
    main()
