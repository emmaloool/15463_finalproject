import numpy as np
from skimage import io

import time, os, glob
from PIL import Image, ImageFont, ImageDraw 

from scipy import ndimage, interpolate
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cv2 import GaussianBlur, line, undistortPoints
from calibrate import calibrateIntrinsic

BASEDIR = "."

'''
Screen backdrops
'''
BACKDROP_DIR = "backdrops"
B_H_DIR, B_V_DIR = ("B_h", "B_v")
BACKDROP_HEIGHT, BACKDROP_WIDTH = (1668, 2388) # 1668 rows, 2388 columns

'''
Mapping image plane pixels (qx,qy) -> backdrop positions (rx,ry)
NOTE: bounds are [min,max)
'''
IMAGE_DIR = "images"
G_BLUR_SIGMA = 0.5

'''
Bounds for cropped image 
Applicable + fixed for both vertical/horizontal directions, this cropping stays constant
'''
IMG = {'cmin':0, 'cmax':2388, 'rmin':0, 'rmax':1668} # TODO adjust 
FRAME = {'h_tmin':0, 'h_tmax':2388, 'v_tmin':0, 'v_tmax':1668}

def gen_backdrop_images():
    backdropdir_path = os.path.join(BASEDIR, BACKDROP_DIR)
    if not os.path.exists(backdropdir_path):
        os.mkdir(backdropdir_path)
    if not os.path.exists(os.path.join(backdropdir_path, B_H_DIR)):
        os.mkdir(os.path.join(backdropdir_path, B_H_DIR))
    if not os.path.exists(os.path.join(backdropdir_path, B_V_DIR)):
        os.mkdir(os.path.join(backdropdir_path, B_V_DIR))
    
    # Using this tutorial to label images: https://towardsdatascience.com/adding-text-on-image-using-python-2f5bf61bf448
    label_font = ImageFont.truetype('font/Nunito-Regular.ttf', 25)

    # Generate B_h (white vertical stripes)
    white_col = np.ones((BACKDROP_HEIGHT, 3))
    for c in range(BACKDROP_WIDTH):
        img_path = os.path.join(backdropdir_path, os.path.join(B_H_DIR, str(c) + ".png"))
        if not os.path.exists(img_path):
            B_h_c = np.zeros((BACKDROP_HEIGHT, BACKDROP_WIDTH, 3))
            B_h_c[:,c] = white_col
            io.imsave(img_path, B_h_c)
        img = Image.open(img_path)
        image_editable = ImageDraw.Draw(img)
        image_editable.text((2225,0), B_H_DIR + "_" + str(c), (255,255,255), font=label_font)
        img.save(img_path)
        
    # Generate B_v (white horizontal stripes)
    white_row = np.ones((BACKDROP_WIDTH, 3))
    for r in range(BACKDROP_HEIGHT):
        img_path = os.path.join(backdropdir_path, os.path.join(B_V_DIR, str(r) + ".png"))
        if not os.path.exists(img_path):
            B_v_r = np.zeros((BACKDROP_HEIGHT, BACKDROP_WIDTH, 3))
            B_v_r[r,:] = white_row
            io.imsave(img_path, B_v_r)
        img = Image.open(img_path)
        image_editable = ImageDraw.Draw(img)
        image_editable.text((2225,0), B_V_DIR + "_" + str(r), (255,255,255), font=label_font)
        img.save(img_path)


'''
Determine a position on the monitor (r_x, r_y) for each pixel in the camera image
that indirectly projects through the center of the pixel
'''
def determine_backdrop_position():
    # test = GaussianBlur(, (0,0), G_BLUR_SIGMA, G_BLUR_SIGMA))
    # test = ndimage.gaussian_filter(rgb2gray(io.imread("./images/test.png")), G_BLUR_SIGMA)
    # io.imsave("./images/result1_ndimage.png", test)

    imgdir_path = os.path.join(BASEDIR, IMAGE_DIR) 
    
    # Parameters for cropped image boundaries
    cmin, cmax, rmin, rmax = (IMG['cmin'], IMG['cmax'], IMG['rmin'], IMG['rmax'])
    IMG_HEIGHT, IMG_WIDTH = (rmax-rmin, cmax-cmin)
    IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH)
    

    '''
        Determining r_x using B_h images
        ----------------------------------------------------------------------------
        1. Generate I(x), the intensity images using backdrop B_h_x.
           To compute for continuous intensities, we apply a Gaussian filter over the intensity image

        2. Approximate the derivative of I(x) at each stripe position by calculating their finite differences:
                            I'(x) = (I(x+1) - I(x-1)) / 2
        
        3. Between each pair of neighboring stripe positions x0 and x0+1, 
        determine the location of the candidate zero-crossing between 

        Assume that the derivative between neighbor stripe positions x0 and x0+1 is linear, 
        construct a line l(x) through the derivatives of x0, x0+1.
        Then solve the zero-crossing as the stripe position x where l(x) = 0

        4. Determine the zero-crossing x_precise representing local maxima as follows:
            - Consider only x_precise where the derivative I'(x) changes from positive to negative 
            - Choose the one that is closest to the discrete stripe position x_discrete, of maximal pixel intensity 
    '''
    def find_r_x():

        h_tmin, h_tmax = (FRAME['h_tmin'], FRAME['h_tmax'])     # [h_tmin, h_tmax)
        NUM_FRAMES = h_tmax - h_tmin
        
        # [1] ------------------------

        I_x = np.zeros((rmax-rmin, cmax-cmin, NUM_FRAMES))
        for t in range(h_tmin, h_tmax):
            img_path = os.path.join(imgdir_path, B_H_DIR, str(t) + ".png")
            img = io.imread(img_path)[rmin:rmax, cmin:cmax]
            I_x[:,:,(t-h_tmin)] = ndimage.gaussian_filter(rgb2gray(img), G_BLUR_SIGMA)


        # [2] ------------------------

        I_prime_x = np.zeros(I_x.shape)

        # At t=h_tmin, h_tmax-1, the derivative estimates are considered invalid (i.e. edge cases for gradient)
        I_prime_x[:,:,0]  = I_x[:,:,0]
        I_prime_x[:,:,-1] = I_x[:,:,-1]

        for t in range(1, NUM_FRAMES-1):
            neighbors = np.dstack(( I_x[:,:,t-1].flatten(), 
                                    I_x[:,:,t  ].flatten(),
                                    I_x[:,:,t+1].flatten() ))
            I_prime_x[:,:,t] = np.squeeze(np.gradient(neighbors, axis=-1))[:,1].reshape(IMG_SHAPE)


        # [3] ------------------------

        # TODO: could move this to be done in the above loop, and instead of doing t,t+1, do t-1,t

        # Calculate candidate position starting at t as that between t and t+1
        # Note: cannot compute zero-crossing candidate for t=h_tmax-h_tmin-1 (edge)
        x_precise_candidates = np.zeros((IMG_HEIGHT, IMG_WIDTH, NUM_FRAMES))
        for t in range(h_tmin, h_tmax-1):

            # Initialize precise candidates for each pixel as INF
            # So values that are unset will be ignored by closest-t tests
            x_precise_t = np.ones((IMG_HEIGHT, IMG_WIDTH)) * np.inf

            I_prime_x_t0, I_prime_x_t1 = (I_prime_x[:,:,t-h_tmin], I_prime_x[:,:,t-h_tmin+1])

            # CASE: Derivative exactly = 0 -> trivially identify it as a zero-crossing
            x_precise_t = np.where(I_prime_x_t0 == 0, t, x_precise_t)

            # CASE: For both neighboring stripe positions, derivative = 0 --> use midpoint of the strip positions
            x_precise_t = np.where((I_prime_x_t0 == 0) & (I_prime_x_t1 == 0), t+0.5, x_precise_t)

            # CASE: I'(x) positive changes -> I'(x+1) negative. Represents local maxima
            x_interpolated_t = t - (I_prime_x_t0 / (I_prime_x_t1 - I_prime_x_t0))
            x_precise_t = np.where((I_prime_x_t0 > 0.0) & (I_prime_x_t1 < 0.0), x_interpolated_t, x_precise_t)
            
            x_precise_candidates[:,:,t-h_tmin] = x_precise_t

        # [4] ------------------------
        
        # Compute the discrete strip position that leads to maximal intensity
        # Note again that we didn't compute the zero-crossing candidate for the last frames
        I_x_max = np.argmax(I_x, axis=-1)
        x_discrete_max = np.repeat((I_x_max)[:,:,np.newaxis], NUM_FRAMES-1, axis=-1)     # titled for comparison

        # Find x_precise nearest to I_x_max
        # Using N-dimension nearest-value finding query described here:
        # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array 
        x_precise_diffs = np.abs(x_precise_candidates - x_discrete_max)
        x_precise_ids = np.nanargmin(x_precise_diffs, axis=-1)
        x_precise = x_precise_candidates.flat[x_precise_ids]

        return x_precise


    '''
        WLOG, the process for determining r_y using B_v is analogous to finding r_x
    '''
    def find_r_y():

        v_tmin, v_tmax = (FRAME['v_tmin'], FRAME['v_tmax'])     # [h_tmin, h_tmax)
        NUM_FRAMES = v_tmax - v_tmin,

        # [1] ------------------------

        I_y = np.zeros((rmax-rmin, cmax-cmin, NUM_FRAMES))
        for t in range(v_tmin, v_tmax):
            img_path = os.path.join(imgdir_path, B_V_DIR, str(t) + ".png")
            img = io.imread(img_path)[rmin:rmax, cmin:cmax]
            I_y[:,:,(t-v_tmin)] = ndimage.gaussian_filter(rgb2gray(img), G_BLUR_SIGMA)


        # [2] ------------------------

        I_prime_y = np.zeros(I_y.shape)
        I_prime_y[:,:,0]  = I_prime_y[:,:,0]
        I_prime_y[:,:,-1] = I_prime_y[:,:,-1]

        for t in range(1, NUM_FRAMES-1):
            neighbors = np.dstack(( I_y[:,:,t-1].flatten(), 
                                    I_y[:,:,t  ].flatten(),
                                    I_y[:,:,t+1].flatten() ))
            I_prime_y[:,:,t] = np.squeeze(np.gradient(neighbors, axis=-1))[:,1].reshape(IMG_SHAPE)


        # [3] ------------------------

        # TODO: could move this to be done in the above loop, and instead of doing t,t+1, do t-1,t
        # Calculate candidate position starting at t as that between t and t+1
        y_precise_candidates = np.zeros((IMG_HEIGHT, IMG_WIDTH, NUM_FRAMES))
        for t in range(v_tmin, v_tmax-1):

            # Initialize precise candidates for each pixel as INF
            # So values that are unset will be ignored by closest-t tests
            y_precise_t = np.ones((IMG_HEIGHT, IMG_WIDTH)) * np.inf

            I_prime_y_t0, I_prime_y_t1 = (I_prime_y[:,:,t-v_tmin], I_prime_y[:,:,(t+1)-v_tmin])

            # CASE: Derivative exactly = 0 -> trivially identify it as a zero-crossing
            y_precise_t = np.where(I_prime_y_t0 == 0, t, y_precise_t)

            # CASE: For both neighboring stripe positions, derivative = 0 --> use midpoint of the strip positions
            y_precise_t = np.where((I_prime_y_t0 == 0) & (I_prime_y_t1 == 0), t+0.5, y_precise_t)

            # CASE: I'(y) positive changes -> I'(y+1) negative. Represents local maxima
            y_interpolated_t = t - (I_prime_y_t0 / (I_prime_y_t1 - I_prime_y_t0))
            y_precise_t = np.where((I_prime_y_t0 > 0.0) & (I_prime_y_t1 < 0.0), y_interpolated_t, y_precise_t)
            
            y_precise_candidates[:,:,t-v_tmin] = y_precise_t


        # [4] ------------------------
        
        # Compute the discrete strip position that leads to maximal intensity
        # Note again that we didn't compute the zero-crossing candidate for the last frames
        I_y_max = np.argmax(I_y, axis=-1)
        y_discrete_max = np.repeat((I_y_max)[:,:,np.newaxis], NUM_FRAMES-1, axis=-1)     # titled for comparison

        # Find y_precise nearest to I_y_max
        # Using N-dimension nearest-value finding query described here:
        # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array 
        y_precise_diffs = np.abs(y_precise_candidates - y_discrete_max)
        y_precise_ids = np.nanargmin(y_precise_diffs, axis=-1)
        y_precise = y_precise_candidates.flat[y_precise_ids]

        return y_precise

    
    r_x, r_y = (find_r_x(), find_r_y())
    return (r_x, r_y)



def main():
    # gen_backdrop_images()
    determine_backdrop_position()

if __name__ == "__main__":
    main()
