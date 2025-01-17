import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import glob

def pdf_multidim(X, mean, cov_inv, cov_det):
    X = X.astype(np.float32)
    d = X.shape[1]
    diff = X - mean
    exponent = -0.5 * np.einsum('ij,ij->i', diff @ cov_inv, diff)
    denom = np.sqrt((2.0 * np.pi)**d * cov_det)
    return (1.0 / denom) * np.exp(exponent)

class GMM():
    def __init__(self, k, samples):
        self.gmm = BayesianGaussianMixture(n_components=k, max_iter=200, covariance_type='full')
        self.gmm.fit(samples)

        self.means = self.gmm.means_
        self.covs = self.gmm.covariances_
        self.weights = self.gmm.weights_
        
        self.k = k
        self.cov_invs = []
        self.cov_dets = []
        for i in range(self.k):
            cov_inv = np.linalg.inv(self.covs[i])
            cov_det = np.linalg.det(self.covs[i])
            self.cov_invs.append(cov_inv)
            self.cov_dets.append(cov_det)

    def pdf(self, X):
        N = X.shape[0]
        p = np.zeros(N, dtype=np.float64)
        for i in range(self.k):
            p_comp = pdf_multidim(X, self.means[i], self.cov_invs[i], self.cov_dets[i])
            p += self.weights[i] * p_comp
        return p

class PDFMultiDim():
    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.cov_inv = np.linalg.inv(self.cov)
        self.cov_det = np.linalg.det(self.cov)

    def pdf(self, X):
        return pdf_multidim(X, self.mean, self.cov_inv, self.cov_det)

def find_mean_cov(img):
    img32 = img.astype(np.float32)
    pixels = img32.reshape(-1, 3)
    mean_hsv = np.mean(pixels, axis=0)
    cov_hsv = np.cov(pixels, rowvar=False)
    return mean_hsv, cov_hsv

def get_samples(path):
    dirs = glob.glob(f"{path}*")
    pixels = np.empty((0, 3), dtype=np.float32)

    for p in dirs:
        img = cv2.imread(p)
        if img is None:
            print(f"Error reading image: {p}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Convert to float32, then flatten
        pixels = np.vstack((pixels, img.reshape(-1, 3).astype(np.float32)))
    return pixels

def main():
    path = sys.argv[1]
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img_bgr is None:
        print(f"Could not read image: {path}")
        return

    # Convert entire image to HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Flatten (H*W, 3)
    H, W, _ = img_hsv.shape
    pixels_hsv = img_hsv.reshape(-1, 3).astype(np.float32)

    # Priors - right now hard coded, should be related to how many total pixels in image
    prior_laser = 0.00001 # 0.001
    prior_bg = 0.99999 # 0.999

    # Build background model
    bg_mean, bg_cov = find_mean_cov(img_hsv)
    pdf_bg = PDFMultiDim(bg_mean, bg_cov)

    # Build laser model from GMM
    laser_samples = get_samples("laser_data/")
    gmm_laser = GMM(k=10, samples=laser_samples)

    # Likelihoods
    likelihood_bg = pdf_bg.pdf(pixels_hsv)
    likelihood_laser = gmm_laser.pdf(pixels_hsv)

    # Combine with priors => posterior
    numerator_laser = likelihood_laser * prior_laser
    numerator_bg = likelihood_bg * prior_bg
    norm_const = numerator_laser + numerator_bg

    # Posterior for laser
    posterior_laser = numerator_laser / norm_const
    # Classify as laser if posterior_laser > posterior_bg => posterior_laser > 0.5
    laser_mask = (posterior_laser > 0.5).reshape(H, W)

    # Color the laser pixels in BGR
    img_bgr[laser_mask] = [128, 0, 0]

    # Plot centroid, indicating laser pixel
    laser_rows, laser_cols = np.where(laser_mask)
    if len(laser_rows) > 0:
        centroid_row = np.mean(laser_rows)
        centroid_col = np.mean(laser_cols)
        print("Laser centroid at (row, col) =", centroid_row, centroid_col)

        # Plot the image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        # Plot the centroid
        plt.scatter(
            [centroid_col], [centroid_row],
            color='red', marker='x', s=50, label='Laser Pixel Mean'
        )
    else:
        print("No laser pixels found; cannot compute centroid.")

    # Show result
    plt.legend()
    plt.imshow(img_rgb)
    plt.title("Laser Pixels (Blue)")
    plt.show()

if __name__ == "__main__":
    main()
