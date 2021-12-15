from typing import NoReturn
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from skimage import io

import time, os, glob, math
from PIL import Image, ImageFont, ImageDraw 

from scipy import ndimage, optimize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from calibrate import calibrateIntrinsic, calibrateExtrinsic, pixel2ray, set_axes_equal
import functools 

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
G_BLUR_SIGMA = 8       # maximum: 10
LOWCONTRAST_MASK = 0.05

'''
Bounds for cropped image 
Applicable + fixed for both vertical/horizontal directions, this cropping stays constant
'''
ROWS, COLS = (275, 317)
H_FRAMES, V_FRAMES = (111, 76)
IMG_OFFSET = {'col':1300, 'row':900}
FRAME_OFFSET = {'h_tmin':700, 'h_tmax':1800, 'v_tmin':550, 'v_tmax':1300}
FRAME_SKIP = 10 # frames skip every 10 

CAPTURE_DIR = "capture"
ROTATED_DIRS = ["0", "neg30", "neg20", "neg10", "10", "20", "30"]
FRONT_DIR, BACK_DIR = ("front", "back")


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
def determine_backdrop_position(view_dir, translate_dir):

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
        
        # [1] ------------------------

        orig_dim = (275, 317)
        I_x = np.zeros((275, 317, H_FRAMES))
        
        for t in range(H_FRAMES):
            imgdir_path = os.path.join(BASEDIR, CAPTURE_DIR, translate_dir, B_H_DIR, view_dir)
            img_path = os.path.join(imgdir_path, "small_" + str(t+1) + ".JPG")
            img = io.imread(img_path)
            I_x[:,:,t] = ndimage.gaussian_filter(rgb2gray(img), G_BLUR_SIGMA)

        # Mask out low-contrast pixels
        I_min = np.min(I_x, axis=2)
        I_max = np.max(I_x, axis=2)
        mask = np.where(np.abs(I_max - I_min) < LOWCONTRAST_MASK, 0, 1)

        # [2] ------------------------

        I_prime_x = np.zeros(I_x.shape)

        # At t=h_tmin, h_tmax-1, the derivative estimates are considered invalid (i.e. edge cases for gradient)
        I_prime_x[:,:,0]  = I_x[:,:,0]
        I_prime_x[:,:,-1] = I_x[:,:,-1]

        for t in range(1, H_FRAMES-1):
            neighbors = np.dstack(( I_x[:,:,t-1].flatten(), 
                                    I_x[:,:,t  ].flatten(),
                                    I_x[:,:,t+1].flatten() ))
            I_prime_x[:,:,t] = np.squeeze(np.gradient(neighbors, axis=-1))[:,1].reshape(orig_dim)

        # [3] ------------------------

        # ************** DISCRETE ZERO-CROSSING IDENTIFICATION  **************
        # x_precise_candidates = np.zeros((ROWS, COLS))
        # for t in range(0, H_FRAMES-1):
        #     I_prime_x_t0, I_prime_x_t1 = (I_prime_x[:,:,t], I_prime_x[:,:,t+1])
        #     x_interpolated_t = (t - (I_prime_x_t0 / (I_prime_x_t1 - I_prime_x_t0))) * 10 + FRAME_OFFSET['h_tmin']
            
        #     x_precise_candidates = np.where((I_prime_x_t0 == 0) & (I_prime_x_t1 == 0), (t * 10) + FRAME_OFFSET['h_tmin'] + 5.0, x_precise_candidates)
        #     x_precise_candidates = np.where((I_prime_x_t0 > 0.0) & (I_prime_x_t1 < 0.0), x_interpolated_t, x_precise_candidates)
        
        # I_min = np.min(I_x, axis=2)
        # I_max = np.max(I_x, axis=2)
        # x_precise_candidates = np.where(np.abs(I_max - I_min) < LOWCONTRAST_MASK, 0, x_precise_candidates)
        
        # plt.imshow(x_precise_candidates, cmap='gray')
        # plt.show()
        # ********************* ********************* *********************

        # Calculate candidate position starting at t as that between t and t+1
        # Note: cannot compute zero-crossing candidate for t=h_tmax-h_tmin-1 (edge)
        x_precise_candidates = np.zeros((ROWS, COLS, H_FRAMES))
        for t in range(0, H_FRAMES-1):
            REAL_T = t*10 + FRAME_OFFSET['h_tmin']      # *** Actual **** frame # t (i.e. B_h_t) should be assigned at this stage!

            x_precise_t = np.zeros((ROWS, COLS))

            I_prime_x_t0, I_prime_x_t1 = (I_prime_x[:,:,t], I_prime_x[:,:,t+1])

            # CASE: Derivative exactly = 0 -> trivially identify it as a zero-crossing
            x_precise_t = np.where(I_prime_x_t0 == 0, REAL_T, x_precise_t)

            # CASE: For both neighboring stripe positions, derivative = 0 --> use midpoint of the strip positions
            # NOTE: Since frames skip every 10 pixels, the midpoints scales too
            x_precise_t = np.where((I_prime_x_t0 == 0) & (I_prime_x_t1 == 0), REAL_T+5.0, x_precise_t)

            # CASE: I'(x) positive changes -> I'(x+1) negative. Represents local maxima
            x_interpolated_t = (t - (I_prime_x_t0 / (I_prime_x_t1 - I_prime_x_t0))) * 10 + FRAME_OFFSET['h_tmin']
            x_precise_t = np.where((I_prime_x_t0 > 0.0) & (I_prime_x_t1 < 0.0), x_interpolated_t, x_precise_t)

            x_precise_candidates[:,:,t] = x_precise_t
            
        # [4] ------------------------
        
        # Compute the discrete strip position that leads to maximal intensity
        # Note again that we didn't compute the zero-crossing candidate for the last frame
        # So that column will not be particularly interesting
        I_x_max = (np.argmax(I_x, axis=-1) * 10) + FRAME_OFFSET['h_tmin']
        x_discrete_max = np.repeat((I_x_max)[:,:,np.newaxis], H_FRAMES, axis=-1)     # tiled across temporal images for comparison

        # Find x_precise nearest to I_x_max
        # Using N-dimension nearest-value finding query described here:
        # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array 
        x_precise_diffs = np.abs(x_precise_candidates - x_discrete_max)
        x_precise_ids = np.argmin(np.ma.masked_invalid(x_precise_diffs), axis=-1)
        x_precise = x_precise_candidates.flat[x_precise_ids]

        x_precise = np.where(mask == 0, 0, x_precise)
        # plt.imshow(x_precise, cmap='gray')
        # plt.savefig(os.path.join(BASEDIR, CAPTURE_DIR, translate_dir + "_" + B_H_DIR + "_" + view_dir))
        # plt.show()

        return x_precise


    '''
        WLOG, the process for determining r_y using B_v is analogous to finding r_x
    '''
    def find_r_y():

        # [1] ------------------------

        orig_dim = (275, 317)
        I_y = np.zeros((275, 317, V_FRAMES))
        
        for t in range(V_FRAMES):
            imgdir_path = os.path.join(BASEDIR, CAPTURE_DIR, translate_dir, B_V_DIR, view_dir)
            img_path = os.path.join(imgdir_path, "small_" + str(t+1) + ".JPG")
            img = io.imread(img_path)
            I_y[:,:,t] = ndimage.gaussian_filter(rgb2gray(img), G_BLUR_SIGMA)

        # Mask out low-contrast pixels
        I_min = np.min(I_y, axis=2)
        I_max = np.max(I_y, axis=2)
        mask = np.where(np.abs(I_max - I_min) < 0.05, 0, 1)

        # [2] ------------------------

        I_prime_y = np.zeros(I_y.shape)

        # At t=h_tmin, h_tmax-1, the derivative estimates are considered invalid (i.e. edge cases for gradient)
        I_prime_y[:,:,0]  = I_y[:,:,0]
        I_prime_y[:,:,-1] = I_y[:,:,-1]

        for t in range(1, V_FRAMES-1):
            neighbors = np.dstack(( I_y[:,:,t-1].flatten(), 
                                    I_y[:,:,t  ].flatten(),
                                    I_y[:,:,t+1].flatten() ))
            I_prime_y[:,:,t] = np.squeeze(np.gradient(neighbors, axis=-1))[:,1].reshape(orig_dim)

        # [3] ------------------------

        # ************** DISCRETE ZERO-CROSSING IDENTIFICATION  **************
        # y_precise_candidates = np.zeros((ROWS, COLS))
        # for t in range(0, V_FRAMES-1):
        #     I_prime_y_t0, I_prime_y_t1 = (I_prime_y[:,:,t], I_prime_y[:,:,t+1])
        #     x_interpolated_t = (t - (I_prime_y_t0 / (I_prime_y_t1 - I_prime_y_t0))) * 10 + FRAME_OFFSET['h_tmin']
            
        #     y_precise_candidates = np.where((I_prime_y_t0 == 0) & (I_prime_y_t1 == 0), (t * 10) + FRAME_OFFSET['h_tmin'] + 5.0, y_precise_candidates)
        #     y_precise_candidates = np.where((I_prime_y_t0 > 0.0) & (I_prime_y_t1 < 0.0), x_interpolated_t, y_precise_candidates)
        
        # I_min = np.min(I_y, axis=2)
        # I_max = np.max(I_y, axis=2)
        # y_precise_candidates = np.where(np.abs(I_max - I_min) < LOWCONTRAST_MASK, 0, y_precise_candidates)
        
        # plt.imshow(y_precise_candidates, cmap='gray')
        # plt.show()
        # ********************* ********************* *********************

        # # Calculate candidate position starting at t as that between t and t+1
        y_precise_candidates = np.zeros((ROWS, COLS, V_FRAMES))
        for t in range(0, V_FRAMES-1):
            REAL_T = t*10 + FRAME_OFFSET['h_tmin']      # *** Actual **** frame # t (i.e. B_h_t) should be assigned at this stage!

            y_precise_t = np.zeros((ROWS, COLS))

            I_prime_y_t0, I_prime_y_t1 = (I_prime_y[:,:,t], I_prime_y[:,:,t+1])

            # CASE: Derivative exactly = 0 -> trivially identify it as a zero-crossing
            y_precise_t = np.where(I_prime_y_t0 == 0, REAL_T, y_precise_t)

            # CASE: For both neighboring stripe positions, derivative = 0 --> use midpoint of the strip positions
            # NOTE: Since frames skip every 10 pixels, the midpoints scales too
            y_precise_t = np.where((I_prime_y_t0 == 0) & (I_prime_y_t1 == 0), REAL_T+5.0, y_precise_t)

            # CASE: I'(y) positive changes -> I'(y+1) negative. Represents local maxima
            y_interpolated_t = (t - (I_prime_y_t0 / (I_prime_y_t1 - I_prime_y_t0))) * 10 + FRAME_OFFSET['h_tmin']
            y_precise_t = np.where((I_prime_y_t0 > 0.0) & (I_prime_y_t1 < 0.0), y_interpolated_t, y_precise_t)

            y_precise_candidates[:,:,t] = y_precise_t

        # [4] ------------------------
        
        # Compute the discrete strip position that leads to maximal intensity
        # Note again that we didn't compute the zero-crossing candidate for the last frames
        I_y_max = (np.argmax(I_y, axis=-1) * 10) + FRAME_OFFSET['h_tmin']
        y_discrete_max = np.repeat((I_y_max)[:,:,np.newaxis], V_FRAMES, axis=-1)     # tiled across temporal images for comparison

        # Find x_precise nearest to I_x_max
        # Using N-dimension nearest-value finding query described here:
        # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array 
        y_precise_diffs = np.abs(y_precise_candidates - y_discrete_max)
        y_precise_ids = np.argmin(np.ma.masked_invalid(y_precise_diffs), axis=-1)
        y_precise = y_precise_candidates.flat[y_precise_ids]

        y_precise = np.where(mask == 0, 0, y_precise)
        # plt.imshow(y_precise, cmap='gray')
        # plt.savefig(os.path.join(BASEDIR, CAPTURE_DIR, translate_dir + "_" + B_V_DIR + "_" + view_dir))
        # plt.show()

        return y_precise

    r_x, r_y = (find_r_x(), find_r_y())
    zeros = np.zeros(r_x.shape)
    
    return np.dstack((r_x, r_y, zeros))


'''
Calibration to simulate multiple rotated camera views
'''
CALIB_DIR = "calibrate"

# Extrinsic calibration parameters
dX = 230.0 #calibration plane length in x direction (in mm)
dY = 160.0 #calibration plane length in y direction (in mm)
dW2 = (8, 8) #window size finding ground plane corners

def calibrate(save_intrinsics=False, save_extrinsics=False): 
    
    '''
        INTRINSIC CALIBRATION
    '''
    def calibrate_intrinsic():
        images = glob.glob(os.path.join(BASEDIR, CALIB_DIR, "*.JPG"))
        mtx, dist = calibrateIntrinsic(images, (6,8), (8,8))
        np.savez(os.path.join(BASEDIR, CALIB_DIR, "intrinsics.npz"), mtx=mtx,
                                                                     dist=dist)
    
    if (save_intrinsics): calibrate_intrinsic()
    with np.load(os.path.join(BASEDIR, CALIB_DIR, "intrinsics.npz")) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
        

    '''
        EXTRINSIC CALIBRATION

        Perform extrinsic calibration against the two screen planes,
        where screen1 refers to the first monitor screen plane, and screen2 refers
        to the screen translated backwards some (fixed) distance on the translation stage.

        So long as we move the screen on the translation stage to fixed positions
        for the front and back screens, the extrinsic calibration process needs only
        to be computed once and shared over the seven different rotated views.

        Since the plane has the same dimensions as the screen for calibration
        (i.e. same 4 corners) as in data capture (i.e. fix front and back position
        to two places where we captured in calibration), we can use the calibration image
        to get extrinsic parameters for the front/back planes.

        Use first image of the front and back screen captures
        WLOG will always collect data for straight-on view (i.e. rotation = 0)

    '''
    def calibrate_extrinsic_translation():
        front_path = os.path.join(BASEDIR, CALIB_DIR, "front.JPG")
        back_path = os.path.join(BASEDIR, CALIB_DIR, "back.JPG")
        tvec_front, rmat_front = calibrateExtrinsic(front_path, mtx, dist, dX, dY)
        tvec_back, rmat_back = calibrateExtrinsic(back_path, mtx, dist, dX, dY)

        np.savez(os.path.join(BASEDIR, CALIB_DIR, "translation_extrinsics.npz"), 
                                                    tvec_front=tvec_front, 
                                                    rmat_front=rmat_front, 
                                                    tvec_back=tvec_back,
                                                    rmat_back=rmat_back)


    # TODO: cleanup organization/naming scheme of rotation calibration images
    # Also make corresponding changes in find_rotation_axis()
    def calibrate_extrinsic_rotation():
        rotation_transforms = {}
        for view in ['neg60', 'neg50', 'neg40', 'neg30', 'neg20', 'neg10', '0', '10', '20', '30', '40', '50', '60']:
            rotated_path = os.path.join(BASEDIR, CALIB_DIR, view + ".JPG")
            tvec, rmat = calibrateExtrinsic(rotated_path, mtx, dist, dX, dY)
            rotation_transforms[view] = {'rmat':rmat, 'tvec':tvec}
        np.savez(os.path.join(BASEDIR, CALIB_DIR, "rotation_extrinsics.npz"), 
                                                    rotation_transforms=rotation_transforms)
        
    if (save_extrinsics): 
        calibrate_extrinsic_translation()
        calibrate_extrinsic_rotation()

    find_rotation_axis()



'''
Reconstruction
'''

# Finds the axis of rotation with respect to the axis of the rotation stage
def find_rotation_axis():
    
    def z0_ray_plane_intersection(o, d):       # in 3D coordinate space of the PLANE. o: 3x1, d: 3x1
        # Use N = [0,0,1], origin = [0,0,0]. c = dot(N, origin: could be any point on the plane though) = 0
        N = np.array([[0,0,1]])
        point = o + (- np.dot(N, o) / np.dot(N, d)) * d
        return point

    def rp_intersect(N, P0, o, d):       # in 3D coordinate space of the CAMERA
        N, P0, o, d = (N.reshape(-1), P0.reshape(-1), o.reshape(-1), d.reshape(-1))    
        c = np.dot(N, P0)
        point = o + ((c - np.dot(N, o)) / np.dot(N, d)) * d
        return point

    def get_rot_plane_eqs():
        planes = []

        with np.load(os.path.join(BASEDIR, CALIB_DIR, "intrinsics.npz")) as X:
                mtx, dist = [X[i] for i in ('mtx', 'dist')]

        for view in ['neg60', 'neg50', 'neg40', 'neg30', 'neg20', 'neg10', '0', '10', '20', '30', '40', '50', '60']:

            rmat, tvec = (rotation_transforms[view]['rmat'], rotation_transforms[view]['tvec'])

            # Cast rays from image corners -> plane (represented in plane coordinate frame)
            a_ray = np.matmul(rmat.T, np.squeeze(200*pixel2ray(np.float32([[0,0]]), mtx, dist)).T)
            b_ray = np.matmul(rmat.T, np.squeeze(200*pixel2ray(np.float32([[BACKDROP_WIDTH, 0]]), mtx, dist)).T)
            c_ray = np.matmul(rmat.T, np.squeeze(200*pixel2ray(np.float32([[0, BACKDROP_HEIGHT]]), mtx, dist)).T)
            d_ray = np.matmul(rmat.T, np.squeeze(200*pixel2ray(np.float32([[BACKDROP_WIDTH, BACKDROP_HEIGHT]]), mtx, dist)).T)
            a_ray, b_ray, c_ray, d_ray = (a_ray.reshape(-1, 1), 
                                            b_ray.reshape(-1, 1),
                                            c_ray.reshape(-1, 1),
                                            d_ray.reshape(-1, 1))
            camera_origin = np.matmul(rmat.T, np.array([[0,0,0]]).T - tvec)
            # Get points of intersection with plane in plane coordinate space
            A,B,C,D = (z0_ray_plane_intersection(camera_origin, a_ray),
                        z0_ray_plane_intersection(camera_origin, b_ray),
                        z0_ray_plane_intersection(camera_origin, c_ray),
                        z0_ray_plane_intersection(camera_origin, d_ray))
            # Convert points back in the camera coordinate system 
            A,B,C,D = (np.matmul(rmat, A) + tvec,
                        np.matmul(rmat, B) + tvec,
                        np.matmul(rmat, C) + tvec,
                        np.matmul(rmat, D) + tvec)
            N = np.cross((B-A).T, (D-B).T)
            N = N / np.linalg.norm(N)
            planes.append((N, A))
        return planes


    # ----------------------------------------------------------------------

    rotation_transforms = dict(np.load(os.path.join(BASEDIR, CALIB_DIR, "rotation_extrinsics.npz"), allow_pickle=True))
    rotation_transforms = dict(rotation_transforms['rotation_transforms'].item())
    # print(rotation_transforms)

    # Obtain coordinates of rotation plane centers in camera coordinates
    X = []
    Y = []
    Z = []
    points = []
    for view in ['neg60', 'neg50', 'neg40', 'neg30', 'neg20', 'neg10', '0', '10', '20', '30', '40', '50', '60']:
        rmat, tvec = (rotation_transforms[view]['rmat'], rotation_transforms[view]['tvec'])
        obj_origin = np.array([0,0,0])
        origin_in_camera = (np.matmul(rmat, obj_origin.reshape(-1, 1)) + tvec).reshape(-1)
        X.append(origin_in_camera[0])
        Y.append(origin_in_camera[1])
        Z.append(origin_in_camera[2])
        points.append((origin_in_camera[0], origin_in_camera[1], origin_in_camera[2]))
        
    # Use least-squares fitting to find the equation of the circle
    # Code obtained from: https://stackoverflow.com/questions/15481242/python-optimize-leastsq-fitting-a-circle-to-3d-set-of-points/15786868
    def find_circle_center():
        def calc_R(xc, yc, zc):
            """ calculate the distance of each 3D points from the center (xc, yc, zc) """
            return np.sqrt((X - xc) ** 2 + (Y - yc) ** 2 + (Z - zc) ** 2)

        def func(c):
            """ calculate the algebraic distance between the 3D points and the mean circle centered at c=(xc, yc, zc) """
            Ri = calc_R(*c)
            return Ri - np.mean(Ri)

        # Determine the least squares circle from these camera coordinate locations
        XM = np.mean(X)
        YM = np.mean(Y)
        ZM = np.mean(Z)
        center_estimate = XM, YM, ZM
        center, ier = optimize.leastsq(func, center_estimate)
        print("CENTER: ", center)


        xc, yc, zc = center
        zc = np.mean(Z)
        Ri       = calc_R(xc, yc, zc)
        R        = np.mean(Ri)
        residu   = np.sum((Ri - R)**2)
        print('R =', R)

        return center
    center = find_circle_center()
    xc,yc,zc = center


    # Identify plane fitting through the circle points, and extract its normal and a point on the plane
    # Code obtained from answer: https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points/20700063#20700063
    def find_circle_plane():
        def plane(x, y, params):
            a = params[0]
            b = params[1]
            c = params[2]
            z = a*x + b*y + c
            return z
        
        def error(params, points):
            result = 0
            for (x,y,z) in points:
                plane_z = plane(x, y, params)
                diff = abs(plane_z - z)
                result += diff**2
            return result

        def cross(a, b):
            return [a[1]*b[2] - a[2]*b[1],
                    a[2]*b[0] - a[0]*b[2],
                    a[0]*b[1] - a[1]*b[0]]

        fun = functools.partial(error, points=points)
        params0 = [0,0,0]
        res = optimize.minimize(fun, params0)

        a = res.x[0]
        b = res.x[1]
        c = res.x[2]
        xs, ys, zs = zip(*points)

        point = np.array([0.0, 0.0, c])
        normal = np.cross(np.array([1,0,a]), np.array([0,1,b]))

        return point, normal, xs, ys, zs
    point, normal, xs, ys, zs = find_circle_plane()
        
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    for view in ['neg60', 'neg50', 'neg40', 'neg30', 'neg20', 'neg10', '0', '10', '20', '30', '40', '50', '60']:
    # for i in range(len(planes)):
        # N, A = planes[i]
        # N = N.flatten()
        # A = A.flatten()

        # Visualize plane in camera coordinate system
        # Based on stack overflow tutorial: https://stackoverflow.com/questions/36060933/matplotlib-plot-a-plane-and-points-in-3d-simultaneously
        # d = -np.dot(N,A)
        # xx, yy = np.meshgrid(np.arange(-200, 200), np.arange(-200, 200))
        # z = (d - N[0]*xx - N[1]*yy) / N[2]
        # ax.plot_surface(xx,yy,z, alpha=0.3, color=[0,1,0])

        rmat, tvec = (rotation_transforms[view]['rmat'], rotation_transforms[view]['tvec'])

        if (view == '60'): marker = 'yo'
        else: marker = 'ro'

        obj_origin = np.array([0,0,0])
        origin_in_camera = (np.matmul(rmat, obj_origin.reshape(-1, 1)) + tvec).reshape(-1)
        # ax.plot(origin_in_camera[0], origin_in_camera[1], origin_in_camera[2], marker)

    axis_pt = center + normal * 10      # arbitrary
    ray = (axis_pt-center) / np.linalg.norm(axis_pt-center)
    intersection = rp_intersect(normal, point, center, ray)

    # ax.plot(intersection[0], intersection[1], intersection[2], 'bo', markersize=8)

    # ax.plot(xc, yc, zc, 'go', markersize=5)
    # # ax.plot(pt[0], pt[1], pt[2], 'bo', markersize=5)
    # ax.set_box_aspect([1,1,1])
    # set_axes_equal(ax)
    
    d = -np.dot(point, normal)
    xx, yy = np.meshgrid(np.arange(-200,100), np.arange(-50, 50))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    # ax.plot_surface(xx, yy, z, alpha=0.4, color=[0,0,1])

    # ax.plot([axis_pt[0], center[0]], [axis_pt[1], center[1]], [axis_pt[2], center[2]], linewidth=2)
    # plt.show()

    axis_direction = (intersection - center) / np.linalg.norm(intersection - center)
    axis_center = intersection

    return (axis_center, axis_direction)




def reconstruction():
    '''
        ###############################################################
        --------------------------- Auxilary --------------------------
        ###############################################################
    '''

    # rmat: 3x3
    # tvec: 3x1, but reshape just in case
    # Returns: 3x1
    def to_camera_point(point, rmat, tvec):
        converted = np.matmul(rmat, point.reshape(-1,1)) + tvec.reshape(-1,1)
        return converted

    # rmat: 3x3
    # tvec: 3x1, but reshape just in case
    # Returns: 3x1
    def to_view_point(point, rmat, tvec):
        converted = np.matmul(rmat.T, point.reshape(-1, 1) - tvec.reshape(-1, 1))
        return converted

    def to_camera_vector(vector, rmat):
        converted = np.matmul(rmat, vector.reshape(-1,1))
        return converted

    def to_view_vector(vector, rmat):
        converted = np.matmul(rmat.T, vector.reshape(-1, 1))
        return converted

    def to_camera(img, rmat, tvec):
        rows, cols, depth = img.shape
        flat = img.reshape(rows*cols, -1)
        rotated = (np.matmul(rmat, flat.T).T).reshape(rows, cols, -1)
        translated = rotated + tvec.reshape(-1)
        return translated

    def to_view(img, rmat, tvec):
        rows, cols, depth = img.shape
        flat = img.reshape(rows*cols, -1)
        translated = flat - tvec.reshape(-1)
        rotated = np.matmul(rmat.T, translated.T).T.reshape(rows, cols, -1)
        return rotated

    def get_c_world(rmat, tvec):
        return np.matmul(-rmat.T, tvec)
    
    # Gets coordinates for each image pixel coordinate
    def get_q_world(dim, rmat, tvec):
        indices = np.float32(np.indices((dim[0],dim[1])).transpose(1,2,0).reshape(dim[0]*dim[1],2))
        undist = cv2.undistortPoints(indices, mtx, dist)
        homogenous = np.squeeze(cv2.convertPointsToHomogeneous(undist))
        inter = np.matmul(rmat.T, (homogenous - tvec.reshape(-1)).T)
        points = (np.matmul(rmat.T, (homogenous - tvec.reshape(-1)).T).T).reshape(dim[0],dim[1],3)
        return points

    # ray = (start, end)
    def gen_samples(segment, limits, n):
        start = segment[0].reshape(-1)
        ray = (segment[1] - segment[0]).reshape(-1)
        ray = ray / np.linalg.norm(ray)
        skip = int(round((limits[1] - limits[0]) / n))
        samples = []
        for scale in range(limits[0], limits[1], skip): 
            print(scale)
            newray = start + ray * scale
            samples.append(newray)
        return samples
    
    # v: 3x1
    # n: 3x1
    def refract(v, n, from_a, to_a):
        v = v.reshape(-1)
        n = n.reshape(-1)

        a = from_a / to_a
        b = np.dot(v, n)
        scale = a*b - math.sqrt(1.0 - ((a**2) * (1.0 - b**2)))
        r = -a*v + scale*n 
        return r

    # l_f : 1x3, f->q (magnitude = distance to image plane from f)
    # r : scalar, IOR
    def compute_nf(c, f, b, l_f, r):
        # Keeping everything a consistent shape
        c = c.reshape(-1)
        f = f.reshape(-1)
        b = b.reshape(-1)
        l_f = l_f.reshape(-1)

        d = np.linalg.norm(l_f)
        o = l_f / d
        i = c - (d*o) - b           # represents lm

        term = i - (np.dot(i, o)*o)
        term = term / np.linalg.norm(term)

        n_f = ( r * np.linalg.norm(i * o) * term ) + ( ((r * np.dot(i, o)) - 1) * o )
        return n_f / np.linalg.norm(n_f)

    # Referenced: https://stackoverflow.com/questions/2259476/rotating-a-point-about-another-point-2d
    def rotate_axis_pt(axis_center, axis_dir, angle, point):
        print("axis_center", axis_center.shape)
        print("axis_dir", axis_dir.shape)
        print("point", point.shape)

        rvec = axis_dir.reshape(-1,1) * np.radians(-angle)
        rmat = cv2.Rodrigues(rvec)[0]

        # Subtract pivot, then perform rotation, before adding the center back
        point = point.reshape(-1,1) - axis_center.reshape(-1,1)
        converted = np.matmul(rmat, point.reshape(-1,1)) + axis_center.reshape(-1,1)
        return converted

    
    # Referenced: https://stackoverflow.com/questions/2259476/rotating-a-point-about-another-point-2d
    def rotate_axis(axis_center, axis_dir, angle, img):
        rvec = axis_dir.reshape(-1,1) * np.radians(-angle)
        rmat = cv2.Rodrigues(rvec)[0]

        # Subtract pivot, then perform rotation, before adding the center back
        orig = img.shape
        flat = img.reshape(orig[0] * orig[1], -1)
        trans_img = flat - axis_center.reshape(-1)
        rotated = np.matmul(rmat, trans_img.T).T
        transformed = (img - axis_center.reshape(-1)).reshape(orig[0], orig[1], -1)
        
        return transformed

    # x, y: (1x3, 1x3)
    def shortest_dist(x, y):
        x1, x2 = x 
        y1, y2 = y
        x1, x2 = (x1.reshape(-1), x2.reshape(-1))
        y1, y2 = (y1.reshape(-1), y2.reshape(-1))

        print("x", (x2 - x1).shape)
        print("y", (y2 - y1).shape)

        cross_diff = np.cross((x2 - x1), (y2 - y1))
        return np.abs(np.dot(y1 - x1, cross_diff)) / np.linalg.norm(cross_diff)

    # Assumes lf, lb, nf are represented in the same coordinate frame
    def evaluate_error(lf, lb, nf):
        f_i, c = (lf[0].reshape(-1,1), lf[1].reshape(-1,1))
        lf_as_vector = c - f_i

        # Derive the implied ray lm
        # lf is traced backward and a ray is refracted
        # Therefore in this backwards sense, the medium changes from air to the material
        lm = (f_i, f_i + refract(lf_as_vector, nf, 1.0, M))

        return (shortest_dist(lm, lb) ** 2)

    # NOTE: lf represented as a directional vector
    def compute_depth(c, surfels):
        c = c.reshape(1, -1)
        points, normals = surfels
        depths = np.zeros(points.shape)
        for i in range(surfels.shape[0]):
            for j in range(surfels.shape[1]):
                depth = (points[i,j,:].reshape(1,-1) - c) / (normals.reshape(1,-1) / np.linalg.norm(normals.reshape(-1)))
                depths[i,j,:] = depth.reshape(1, -1)
        return depths

    '''
        ###############################################################
        ---------------------------- Setup ----------------------------
        ###############################################################
    '''

    # Load intrinsic parameters
    with np.load(os.path.join(BASEDIR, CALIB_DIR, "intrinsics.npz")) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]

    # Load translations transforms for translated views
    # We'll use them to express each rotated view's front and back views in the same coordinate frame
    with np.load(os.path.join(BASEDIR, CALIB_DIR, "translation_extrinsics.npz")) as X:
        tvec_front, rmat_front, tvec_back, rmat_back = [X[i] for i in ('tvec_front', 'rmat_front', 'tvec_back', 'rmat_back')]

    '''
        ###############################################################
        ----------------------- Reconstruction ------------------------
          Perform depth sampling with respect to the reference camera
        ###############################################################
    '''
    
    '''
        The glass is not far from the translation stage (separated by 10-30cm)
        So we can conservatively choose samples between 0cm beind/40cm in front true plane position
        for the position where light first enters the object from the back

        Similarly, for the camera, it's separated from the rotation stage by 50cm.
        So we give more weight to sampling behind the rotation stage.
    '''

    B_LIMITS = (0, 40)
    F_LIMITS = (-20, 20)

    N = 1  # TODO: tune
    M = 1.5 # TODO: restructure to estimate depth instead of using this constant

    # ---------- Reference camera: rotated view 0 degrees, expressed in front plane coordinate frame ----------
    # For a given camera, where q = (r,c), Lb = L(q) = (r_back(r,c) r_front(r,c))
    # We will commonly express computations in the coordinate space of the front frame
    # So we'll need to convert the back backdrop positions into the coordinate space of the front

    # ref_r_front = determine_backdrop_position("0", FRONT_DIR) 
    # ref_r_back = to_view(to_camera(determine_backdrop_position("0", BACK_DIR), rmat_back, tvec_back), rmat_front, tvec_front)

    ref_rmat, ref_tvec = (rmat_front, tvec_front)
    ref_c = np.array([[0,0,0]]) #get_c_world(ref_rmat, ref_tvec)
    ref_Q = to_camera(get_q_world((1668, 2388), ref_rmat, ref_tvec), ref_rmat, ref_tvec)

    '''  Use rotated views to recover the object's axis of rotation '''
    axis_center, axis_dir = find_rotation_axis()
    print("############ ROTATION AXIS = ", axis_center, axis_dir, "############")

    # Pre-compute L_v(q) for the remaining cameras (validation camera views v)
    # We also need image pixels -> world positions for each of the views
    VIEW_WORLD_PTS = {}
    for view in [[0, "0"], [-10, "neg10"], [-20, "neg20"], [-30, "neg30"], [10, "10"], [20, "20"], [30, "30"]]:
        angle, dir = view

        view_front = to_camera(determine_backdrop_position(dir, FRONT_DIR), rmat_front, tvec_front)
        view_back  = to_camera(determine_backdrop_position(dir, BACK_DIR), rmat_back, tvec_back)
        
        world_front = rotate_axis(axis_center, axis_dir, angle, view_front)
        world_back = rotate_axis(axis_center, axis_dir, angle,  view_back)

        VIEW_WORLD_PTS[dir] = {"front":world_front, "back":world_back}


    '''
        Main depth sampling algorithm
        For each pixel in the image, we aspire to get the best surfel (point-normal)
        combination that will allow us to get depth of the point on the object that projects through the pixel
    '''
    def depth_sample(): 

        # Store a surfel for each pixel in the image
        surfel_points  = np.zeros((ROWS, COLS, 3))
        surfel_normals = np.zeros((ROWS, COLS, 3))

        # Iterate over pixels in image
        for y in range(ROWS): #for img_y in range(ROWS):
            for x in range(COLS):   #for img_x in range(COLS):

                # NOTE: Rays are represented as pairs of points (start, end)
                # Lb_ref = (ref_r_back[y,x], ref_r_front[y,x])    # Lb1 of the reference camera
                # Lf     = (ref_Q[y,x], ref_c.reshape(-1))

                Lb_ref = (VIEW_WORLD_PTS["0"]["back"][y,x], VIEW_WORLD_PTS["0"]["front"][y,x])    # Lb1 of the reference camera
                Lf     = (ref_Q[y,x], ref_c.reshape(-1))

                b_samples = gen_samples(Lb_ref, B_LIMITS, N)
                f_samples = gen_samples(Lf, F_LIMITS, N)

                '''
                    Perform 3D search along samples of b,f positions
                    Iteratively keep track of best error/surfel pair, where surfel = (f_i, nf)
                    NOTE: will store f_i, nf in the coordinates of the reference coordinate frame
                '''
                best_error = np.inf
                best_surfel = (np.array(([0,0,0])), np.array(([0,0,0])))

                for j in range(len(b_samples)):
                    for i in range(len(f_samples)):
                        print("----------------- x", x, "y", y, "i", i, "j", j, "-----------------")

                        b_j, f_i = (b_samples[j], f_samples[i])

                        # Determine the normal n_f in the reference view's coordinate frame
                        nf = compute_nf(ref_c, f_i, b_j, (Lf[1] - Lf[0]), M)

                        '''
                            Evaluate light path consistency error for all cameras
                            NOTE: We omit the reference camera from consideration
                        '''
                        E_s_ij = 0      # error across all validation view cameras

                        for angle, view in [[-10, "neg10"], [-20, "neg20"], [-30, "neg30"], [10, "10"], [20, "20"], [30, "30"]]:
                            if (angle == 0): continue;   # skip reference camera

                            view_c = rotate_axis_pt(axis_center, axis_dir, angle, ref_c.reshape(-1,1))

                            # Project the surfel (specifically point f but from the perspective of the validation camera) into the image plane to get image point qc
                            q_c_x, q_c_y = (cv2.projectPoints(view_c.astype(np.float32), cv2.Rodrigues(ref_rmat)[0], ref_tvec, mtx, dist)[0].astype(int).reshape(1,2))[0]
                            # Need to acount for cropped image coordinates
                            q_c_x = np.clip(q_c_x - IMG_OFFSET['col'], 0, COLS-1)
                            q_c_y = np.clip(q_c_y - IMG_OFFSET['row'], 0, ROWS-1)

                            # Then we can use our per-view backdrop lookup map to get the correct positions of the points in that view's camera coordinate frame

                            # Query the initial light ray for this view
                            Lb_j_c = (VIEW_WORLD_PTS[view]["back"][q_c_y, q_c_x], VIEW_WORLD_PTS[view]["front"][q_c_y, q_c_x])

                            # Query the final ray
                            Lf_i_c = (f_i, view_c) # (view_Q[q_c_y, q_c_x], view_c)

                            # Get nf expressed in the validation camera's coordinate frame
                            view_nf = rotate_axis_pt(axis_center, axis_dir, angle, nf)

                            # Evaluate the reconstruction error e_c_s_ij
                            e_c_s_ij = evaluate_error(Lf_i_c, Lb_j_c, view_nf)
                            E_s_ij += e_c_s_ij
                        
                        if (E_s_ij < best_error):
                            best_surfel = (f_i, nf)

                        # print(" ENDS HERE ")
                        # break;
                
                surfel_points[y,x,:] = best_surfel[0].reshape(-1,3)
                surfel_points[y,x,:] = best_surfel[1].reshape(-1,3)

        # np.savez("surfels.npz", points=surfel_points, normals=surfel_normals)    
        return surfel_points, surfel_normals

    surfels = depth_sample()

    depths = compute_depth(ref_c, surfels)
    return depths


def main():
    start = time.time()

    # gen_backdrop_images()
    # calibrate()

    reconstruction()

    end = time.time()
    print("*** Script duration: " + str(end - start) + " ***")

if __name__ == "__main__":
    main()
