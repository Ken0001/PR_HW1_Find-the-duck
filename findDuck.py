import cv2 as cv
import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt


duck_body=[]
non_duck_body=[]
for i in range(1,23):
    duck_body.append("duck_body ("+str(i)+").png")
for i in range(1,19):
    non_duck_body.append("non_duck_body ("+str(i)+").png")

# read duck & non ducck
def readFiles(files):
    pixels = np.empty((0,3))
    for f in files:
        data = cv.imread("dataset/"+f)
        pixels = np.vstack([pixels, np.reshape(data, (-1,3))])
    # print(pixels)
    return pixels
duck_pixels = readFiles(duck_body)
non_duck_pixels = readFiles(non_duck_body)
# print('Duck Pixels Data:', duck_pixels.shape)
# print('Non-Duck Pixels Data:', non_duck_pixels.shape)

# Likelihood model (mean vector & covariance matrix)
duck_mean = np.mean(duck_pixels, 0)
duck_cov = np.cov(duck_pixels.T)
non_duck_mean = np.mean(non_duck_pixels, 0)
non_duck_cov = np.cov(non_duck_pixels.T)

# Compute the p(x|Duck) for all trained duck & nonduck pixels
duck_given_duck = mvn.pdf(duck_pixels, duck_mean, duck_cov)
non_duck_given_duck = mvn.pdf(non_duck_pixels, duck_mean, duck_cov)

# Compute the p(x|nonDuck) for all trained duck & nonduck pixels
duck_given_non_duck = mvn.pdf(duck_pixels, non_duck_mean, non_duck_cov)
non_duck_given_non_duck = mvn.pdf(non_duck_pixels, non_duck_mean, non_duck_cov)

# Compute the likelihood ratio p(x|Duck)/p(x|Non-Duck) for all duck & nonduck pixels
likelihood_ratio1 = duck_given_duck/duck_given_non_duck
likelihood_ratio2 = non_duck_given_duck/non_duck_given_non_duck
# for storing the accuracy rates of different theta values
hit_rates = [] 
theta_range = np.arange(0.8, 200.0, 0.5) 
# Count duck hit or non duck hit
for theta in theta_range:
    num_duck_hit = np.sum(likelihood_ratio1>theta)   
    num_non_duck_hit = np.sum(likelihood_ratio2<=theta) 
    rate = (num_duck_hit + num_non_duck_hit)/(len(duck_pixels)+len(non_duck_pixels)) # accuracy rate
    print('theta = %.2f hit_rate = %.5f'%(theta, rate))
    hit_rates.append(rate) 

best_theta = theta_range[np.argmax(hit_rates)]
print(best_theta)
plt.plot(theta_range, hit_rates) 
plt.show()


all_pixels = readFiles(['full_duck.jpg'])
duck_probs = mvn.pdf(all_pixels, duck_mean, duck_cov)
non_duck_probs = mvn.pdf(all_pixels, non_duck_mean, non_duck_cov)
duck_mask = (duck_probs/non_duck_probs)>best_theta
# print(duck_mask.shape)
duck_mask = np.reshape(duck_mask, (13816, 5946))
# print(duck_mask.shape)
# Draw result
full_img = cv.imread('full_duck.jpg')
duck_img = np.zeros_like(full_img)
for ch in range(3):
    duck_img[:,:,ch] = full_img[:,:,ch]*duck_mask
cv.imwrite('duck_only.jpg', duck_img)