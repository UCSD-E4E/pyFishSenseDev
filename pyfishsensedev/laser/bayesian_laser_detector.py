import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

class PDFOneDim():
    def __init__(self, meu, sigma):
        self.meu = meu
        self.sigma = sigma
    def __call__(self, x):
        coeff = 1.0 / (math.sqrt(2.0 * math.pi) * self.sigma)
        exponent = -((x - self.meu) ** 2) / (2.0 * (self.sigma ** 2))
        return coeff * math.exp(exponent)

class PDFMultiDim():
    def __init__(self, meu, cov):
        self.meu = np.asarray(meu)
        self.cov = np.asarray(cov)
        self.cov_det = np.linalg.det(self.cov)
        self.cov_inv = np.linalg.inv(self.cov)
        self.denom = np.sqrt((2 * np.pi)**3 * self.cov_det)
    def __call__(self, x):
        x = np.asarray(x)
        diff = x - self.meu
        exponent = -0.5 * diff.T @ self.cov_inv @ diff
        return (1.0 / self.denom) * np.exp(exponent)

def find_mean_cov(img):
    img32 = img.astype(np.float32)
    pixels = img32.reshape(-1, 3)
    mean_hsv = np.mean(pixels, axis=0) 
    cov_hsv = np.cov(pixels, rowvar=False)
    return mean_hsv, cov_hsv

def main():
    path = sys.argv[1]
    img_bgr = cv2.imread(path) # cv2's default color space is BGR
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    laser_pixels = []

    # Priors
    prior_laser = 0.001 # 0.001
    prior_bg = 0.999 # 0.999

    # Likelihoods
    laser_mean = [138.60448, 73.84311, 250.71721]
    laser_cov = [[4127.75375298, 1053.89700979, -111.98989248],
                [1053.89700979, 523.51362831, -53.9500694],
                [-111.98989248, -53.9500694, 31.80453888]]
    bg_mean, bg_cov = find_mean_cov(img_hsv)
    p_laser = PDFMultiDim(laser_mean, laser_cov)
    #p_bg = PDFOneDim(100, 20)
    p_bg = PDFMultiDim(bg_mean, bg_cov)
    # for each pixel
    for r in range(0, len(img_hsv)):
        for c in range(0, len(img_hsv[r])):
            likelihood_laser = p_laser(img_hsv[r][c].astype(int))
            #likelihood_bg = p_bg(int(img_bgr[r][c][2])) # only the red channel
            likelihood_bg = p_bg(img_hsv[r][c].astype(int))
            # Posterior
            norm_const = likelihood_laser * prior_laser + likelihood_bg * prior_bg
            posterior_laser = (likelihood_laser * prior_laser) / norm_const
            posterior_bg = (likelihood_bg * prior_bg) / norm_const
            #print(f"Laser posterior: {posterior_laser} | BG posterior: {posterior_bg}")
            if posterior_laser > posterior_bg:
                laser_pixels.append([r, c])
    
    print(laser_pixels)

    for pair in laser_pixels:
        r = pair[0]
        c = pair[1]
        img_bgr[r][c] = [128,0,0]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

    #############################################

    # test pdf
    # avg = 225
    # std_dev = 10
    # pdf = PDFOneDim(avg, std_dev)

    # x = [val for val in range(0,255)]
    # y = [pdf(val) for val in x]

    # data_x = reds.ravel()
    # data_y = [pdf(int(val)) for val in data_x]

    # plt.plot(x, y)
    # plt.scatter(data_x, data_y, color="hotpink", label="Real data")
    # plt.xlabel("Red channel")
    # plt.ylabel("Probability density")
    # plt.legend()
    # plt.title("1-D Gaussian PDF (laser class)")
    # plt.show()

if __name__ == "__main__":
    main()