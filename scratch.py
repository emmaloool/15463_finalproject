import numpy as np
from skimage import io
import math

import time, os, glob, subprocess, shlex
from PIL import Image, ImageFont, ImageDraw 

from scipy import ndimage, interpolate
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cv2 import GaussianBlur, line, undistortPoints
from calibrate import calibrateIntrinsic


def blah():
    blah = np.array([[1,1,1,1],
                     [1,1,1,1],
                     [1,1,1,1]])

    meh = np.array([[1,2,3,4],
                    [4,5,6,7],
                    [7,8,9,10]])

    bleh = np.array([[1,4,9,16],
                     [25,36,49,64],
                     [81,100,121,132]])

    sotrue = np.array([[10,20,30,40],
                    [50,60,70,80],
                    [90,100,110,120]])


    # -------------------------------------------------------------------
    #                   figuring out gradient across z-axis
    # -------------------------------------------------------------------

    blah = blah.flatten()
    meh = meh.flatten()
    bleh = bleh.flatten()
    sotrue = sotrue.flatten()

    e = np.dstack((blah, meh, bleh, sotrue))
    cool = np.dstack((blah, meh, bleh))
    other = np.dstack((meh, bleh, sotrue))

    # # I' for meh
    print(np.squeeze(np.gradient(cool, axis=-1))[:,1].reshape((3,4)))
    print(np.gradient(e, axis=-1)[0][:,1].reshape((3,4)))
    print(((bleh - blah)/2).reshape((3,4)))
    

    # # I' for bleh
    # print("\n#############\n")
    # print(np.squeeze(np.gradient(other, axis=-1))[:,1].reshape((3,4)))
    # print(np.squeeze(np.gradient(e, axis=-1))[:,2].reshape((3,4)))
    # print(((sotrue - meh)/2).reshape((3,4)))

    # -------------------------------------------------------------------
    #                   figuring out maximum across z-axis
    # -------------------------------------------------------------------
 
    # blah = np.array([[1,2,3,4],
    #                 [4,5,6,7],
    #                 [100,100,9,10]])

    # meh = np.array([[1,1,1,1],
    #                 [1,1,1,1],
    #                 [81,100,121,132]])


    # bleh = np.array([[1,4,9,16],
    #                  [25,36,49,64],
    #                  [1,1,1,1]])

    # sotrue = np.array([[10,20,30,40],
    #                 [50,60,70,80],
    #                 [90,100,121,133]])

    # print(np.argmax(np.dstack((blah, meh, bleh, sotrue)), axis=-1))
    # print(np.max(np.dstack((blah, meh, bleh, sotrue)), axis=-1))

    # -------------------------------------------------------------------
    #                   figuring out closest value
    # -------------------------------------------------------------------

    # A = np.ones((3,4)) 
    # B = np.ones((3,4)) * 3
    # C = np.ones((3,4)) * 10
    # D = np.ones((3,4)) * -np.inf
    # X = np.dstack((A,B,C,D))

    # target = np.array([[0,3,12,18],
    #                    [51,10,3,1],
    #                    [-100, 10000, 21, 16]])#[-100, np.inf, np.nan, -np.inf]])
    # repeat_target = np.repeat(target[:,:,np.newaxis], 4, axis=-1)
    # # print(repeat_target)
    # print(repeat_target.shape)

    # ids = (np.abs(X - repeat_target))
    # print(ids)
    # ids = np.nanargmin(ids, axis=-1)
    # print(ids)
    # values = X.flat[ids]
    # print(values)


    # blah = np.array([[1,2,3,-1,-1],[7,8,9,-3,-3]])
    # zeros = np.zeros(blah.shape)
    # print(cool.shape)
    # print(cool)
    # print(cool.T.shape)
    # print(cool.T)
    # supercool = cool.flatten().reshape(cool.shape[0] * cool.shape[1],3)
    # # print(supercool.shape)
    # # print(supercool)


    # print(np.array([1,2,3]).reshape(-1))
    # print(np.array([[1],[2],[3]]).reshape(-1))
    # print(np.float32([10,20,30]))

    # example = np.array([[-1,0,0], [0,1,0], [0,0,-1]])
    # # example = np.dstack((example, example, example))


    # meh = np.array([[5,22,33,44,5], [7,7,7,7,7], [7,7,7,7,7], [5,22,33,44,5]])
    # zeros = np.zeros(meh.shape)
    # cool = np.dstack((meh, zeros,  np.ones(meh.shape) * 11))
    # # print("cool shape:" ,cool.shape, cool)

    # flat_cool = cool.reshape(5*4, -1)
    # print("flat cool shape:", flat_cool.shape, flat_cool)

    # result = np.zeros((4,5,3))
    # for r in range(4):
    #     for c in range(5):
    #         result[r,c,:] = np.matmul(example, cool[r,c,:].reshape(-1,1)).reshape(-1,3)
    # print(result) 

    # print("example shape:" ,example.shape, example)
    # print(cool - np.array([-10, -2, 0]))
    # print("")

    # print(np.matmul(example, flat_cool.T).T.reshape(4,5, -1))
    # meh = np.indices((5,4)).transpose(1,2,0)
    # meh = meh.reshape(5*4, 2)
    # print(meh)


def main():

    blah()

    # old_base = "./old_capture/"
    # NEW = "../../../../capture"

    # convert -crop +1300+900 -crop -1375-825 1.jpg out.jpg

    # subprocess.call(shlex.split("cd old_capture/back/B_v/30/"), shell=True)
    # subprocess.call(shlex.split("pwd"), shell=True, cwd="./old_capture/back/B_v/30/")
    # subprocess.call(shlex.split("convert -crop +1300+900 1.JPG output.JPG"), shell=True, cwd="./old_capture/back/B_v/30/")

    
    # for trans in ["front", "back"]:
    #     for direction in ["B_h", "B_v"]:
    #         for rotation in ["0", "-30", "-20", "-10", "10", "20", "30"]:
    #             old_path = os.path.join(old_base, trans, direction, rotation)
    #             new_path = os.path.join(NEW, trans, direction, rotation)

    #             # if not os.path.exists(new_path):
    #             #     os.mkdir(new_path)

    #             if (direction == "B_v"):
    #                 for i in range(1,2):
    #                     old_img = str(i) + ".JPG"
    #                     new_img = os.path.join(new_path, str(i-1) + ".JPG")
    #             if (direction == "B_h"):
    #                 for i in range(1,2):
    #                     old_img = str(i) + ".JPG"
    #                     new_img = os.path.join(new_path, str(i-1) + ".JPG")

    #             cmd = "convert -crop +1300+900 -crop -1375-825 " + old_img + " " + new_img
    #             # subprocess.list2cmdline

    #             print(cmd)
                
    #             # subprocess.call(shlex.split("convert -crop +1300+900 -crop -1375-825 1.JPG ../../../../capture/back/B_v/30/0.JPG"), shell=True)
    #             # subprocess.call(shlex.split(cmd), shell=True, cwd=old_path)
    #             # subprocess.call(shlex.split("cd ~/git/comp-photo/final/"), shell=True)
    #             # blah = subprocess.call(["convert", "-crop", "+1300+900", "-crop", "-1375-825", old_img, new_img], shell=True, cwd=old_path)
    #             # blah = subprocess.call(["cat", str(i)+".JPG"], shell=True, cwd=old_path)

            


if __name__ == "__main__":
    main()