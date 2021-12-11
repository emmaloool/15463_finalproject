import numpy as np
from skimage import io

import time, os, glob
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



'''
Calibration to simulate multiple rotated camera views
'''
CALIB_DIR = "calibrate"
CAPTURE_DIR = "capture"
ROTATED_DIRS = ["0", "-10", "-20", "-30", "10", "20", "30"]
FRONT_DIR, BACK_DIR = ("front", "back")

# Extrinsic calibration parameters
dX = 230.0 #calibration plane length in x direction (in mm)
dY = 160.0 #calibration plane length in y direction (in mm)
dW2 = (8, 8) #window size finding ground plane corners

def calibrate(save_intrinsics=True, save_extrinsics=True): 
    
    '''
        INTRINSIC CALIBRATION
    '''
    def calibrate_intrinsic():
        images = glob.glob(os.path.join(BASEDIR, CALIB_DIR, "*.JPG"))
        mtx, dist = calibrateIntrinsic(images, (6,8), (8,8))
        np.savez(os.path.join(BASEDIR, CALIB_DIR, "intrinsics.npz"), mtx=mtx,
                                                                     dist=dist)
    
    # if (save_intrinsics): calibrate_intrinsic()
    with np.load(os.path.join(BASEDIR, CALIB_DIR, "intrinsics.npz")) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
        

    '''
        EXTRINSIC CALIBRATION

        Perform extrinsic calibration against the two screen planes,
        where screen1 refers to the first monitor screen plane, and screen2 refers
        to the screen translated backwards some (fixed) distance on the translation stage.

        So long as we move the screen on the translation stage to fixed positions
        for the front and back screens, the extrinsic calibration process needs only
        to be computed once and shared over the seven different rotated views

    '''
    def calibrate_extrinsic_translation(screen1_path, screen2_path):
        tvec_front, rmat_front = calibrateExtrinsic(screen1_path, mtx, dist, dX, dY)
        tvec_v_back, rmat_v_back = calibrateExtrinsic(screen2_path, mtx, dist, dX, dY)

        ext_out = {"tvec_front":tvec_front, "rmat_front":rmat_front, "tvec_v_back":tvec_v_back, "rmat_v_back":rmat_v_back}
        np.savez(os.path.join(BASEDIR, CALIB_DIR, "extrinsics.npz"), tvec_front=tvec_front, 
                                                                     rmat_front=rmat_front, 
                                                                     tvec_v_back=tvec_v_back,
                                                                     rmat_v_back=rmat_v_back)

    def calibrate_extrinsic_rotation():

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

        def get_planes():
            transforms = []
            checkboard_centers = []    # in camera coordinate space
            planes = []
            for i in range(1,16):
                rotated_path = os.path.join(BASEDIR, CALIB_DIR, str(i) + ".JPG")
                I = rgb2gray(io.imread(rotated_path))

                tvec, rmat = calibrateExtrinsic(rotated_path, mtx, dist, dX, dY)
                transforms.append({'rmat':rmat, 'tvec':tvec})

                # Cast rays from image corners -> plane (represented in plane coordinate frame)
                a_ray = np.matmul(rmat.T, np.squeeze(200*pixel2ray(np.float32([[0,0]]), mtx, dist)).T)
                b_ray = np.matmul(rmat.T, np.squeeze(200*pixel2ray(np.float32([[I.shape[1], 0]]), mtx, dist)).T)
                c_ray = np.matmul(rmat.T, np.squeeze(200*pixel2ray(np.float32([[0, I.shape[0]]]), mtx, dist)).T)
                d_ray = np.matmul(rmat.T, np.squeeze(200*pixel2ray(np.float32([[I.shape[1], I.shape[0]]]), mtx, dist)).T)
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
        # get_planes()

        planes = np.load('planes.npz', allow_pickle=True)['planes']
        transforms = np.load('transforms.npz', allow_pickle=True)['transforms']

        X = []
        Y = []
        Z = []
        points = []
        for i in range(len(transforms)):
            rmat, tvec = (transforms[i]['rmat'], transforms[i]['tvec'])
            obj_origin = np.array([0,0,0])
            origin_in_camera = (np.matmul(rmat, obj_origin.reshape(-1, 1)) + tvec).reshape(-1)
            X.append(origin_in_camera[0])
            Y.append(origin_in_camera[1])
            Z.append(origin_in_camera[2])
            points.append((origin_in_camera[0], origin_in_camera[1], origin_in_camera[2]))
        
        # Use least-squares fitting to find the equation of the circle
        # Code obtained from: https://stackoverflow.com/questions/15481242/python-optimize-leastsq-fitting-a-circle-to-3d-set-of-points/15786868
        # TODO: Broken
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
            Ri       = calc_R(xc, yc, zc)
            R        = np.mean(Ri)
            residu   = np.sum((Ri - R)**2)
            print('R =', R)

            return center
        xc,yc,zc = find_circle_center()

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
        
        def get_axis():
            # Axis is centered at the center of the circle
            # and extends outward from the plane of the circle
            #


        def visualize_planes():
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
            for i in range(len(planes)):
                N, A = planes[i]
                N = N.flatten()
                A = A.flatten()

                # Visualize plane in camera coordinate system
                # Based on stack overflow tutorial: https://stackoverflow.com/questions/36060933/matplotlib-plot-a-plane-and-points-in-3d-simultaneously
                # d = -np.dot(N,A)
                # xx, yy = np.meshgrid(np.arange(-200, 200), np.arange(-200, 200))
                # z = (d - N[0]*xx - N[1]*yy) / N[2]
                # ax.plot_surface(xx,yy,z, alpha=0.3, color=[0,1,0])

                rmat, tvec = (transforms[i]['rmat'], transforms[i]['tvec'])

                obj_origin = np.array([0,0,0])
                origin_in_camera = (np.matmul(rmat, obj_origin.reshape(-1, 1)) + tvec).reshape(-1)
                ax.plot(origin_in_camera[0], origin_in_camera[1], origin_in_camera[2], 'ro')

            ax.plot(xc, yc, zc, 'go', markersize=5)
            # ax.plot(pt[0], pt[1], pt[2], 'bo', markersize=5)
            ax.set_box_aspect([1,1,1])
            set_axes_equal(ax)
            
            d = -np.dot(point, normal)
            xx, yy = np.meshgrid(np.arange(-150,50), np.arange(-100, 100))
            z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            ax.plot_surface(xx, yy, z, alpha=0.2, color=[1,0,0])


            plt.show()
        visualize_planes()

        



    # -------------------- Translation stage --------------------
    # Use first image of the front and back screen captures
    # WLOG will always collect data for straight-on view (i.e. rotation = 0)
    front_path = os.path.join(BASEDIR, CAPTURE_DIR, ROTATED_DIRS[0], FRONT_DIR)
    back_path = os.path.join(BASEDIR, CAPTURE_DIR, ROTATED_DIRS[0], BACK_DIR)
    # if (save_extrinsics): calibrate_extrinsic_translation(front_path, back_path)




    # -------------------- Rotation stage --------------------
    # Find the axis of rotation of the rotation stage using the least squares circle
    # of the checkerboard origins of the rotated views   of the calibration plane on the rotation stage 
    if (save_extrinsics): calibrate_extrinsic_rotation()



'''
Reconstruction
'''
def reconstruction():
    with np.load(os.path.join(BASEDIR, CALIB_DIR, "intrinsics.npz")) as X:
        mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

    '''
        For each camera view:
        - rmat, tvec: extrinsic parameters from plane
    '''

    def get_c_world(rmat, tvec):
        return np.matmul(-rmat.T, tvec)

    # x, y: coordinates (COLUMN, ROW)!
    # TODO: apply rotation to get back into world coordinates
    def get_q_world(x, y, rmat, tvec):
        undist = cv2.undistortPoints(np.float32([[x, y]]), mtx, dist)
        homogenous = cv2.convertPointsToHomogeneous(undist)
        point = np.matmul(rmat.T, homogenous - tvec)
        point = point[0,:].reshape(-1, 1)
        return point


    # l_f : 1x3, f->q (magnitude = distance to image plane from f)
    # r : scalar, IOR
    def compute_nf(c, f, b, l_f, r):
        d = np.linalg.norm(l_f)
        o = l_f / d
        i = c - (d*o) - b           # represents lm

        term = i - (np.dot(i, o)*o)
        term = term / np.linalg.norm(term)

        n_f = ( r * np.linalg.norm(i * o) * term ) + ( ((r * np.dot(i, o)) - 1) * o )
        return n_f



def main():
    # gen_backdrop_images()

    calibrate()

    # determine_backdrop_position()

if __name__ == "__main__":
    main()
