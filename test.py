import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt
# ============ SSTV ============
from lib.sstv import cv_gen_trajectory

if __name__ == "__main__":
    image = io.imread('img857.jpg')

    boxes = np.array(
        [[[128, 196, 340, 351, 237, 167],
        [215, 112,  62, 135, 173, 279]],
        [[349, 459, 510, 544, 482, 351],
        [505, 456, 370, 379, 495, 543]]]
    )
    boxes = np.moveaxis(boxes,-1,-2)

    seed = np.random

    # trajectory parameters
    maxstep = 10
    # [-1.1*270,1.1*270]
    rotate = 270 *(seed.random()-0.5)*2.2
    # [-1.1*200,1.1*200]
    shiftx = 200 *(seed.random()-0.5)*2.2
    shifty = 200 *(seed.random()-0.5)*2.2
    # [0.9*2,1.1*2]
    scalex = 2 *((seed.random()-0.5)*0.2+1)
    scaley = 2 *((seed.random()-0.5)*0.2+1)
    
    image_list,poly_xy_list,Ms,step_state_dict,blur_stepi = cv_gen_trajectory(
        image,maxstep,boxes,
        rotate=rotate,shift=(shifty,shiftx),scale=(scaley,scalex),
        blur=True,blur_rate=0.6,blur_ksize=15,blur_intensity=0.2,
        blur_return_stepi=True,return_states=True,
        )

    color = (0,255,0)
    thickness=3
    fig,axs = plt.subplots(2,5)
    for ax,img,bxs in zip(axs[0],image_list[:5],poly_xy_list[:5]):
        for bx in bxs:
            cv2.polylines(img,[bx.reshape((-1,1,2)).astype(np.int32)],True,color,thickness=thickness)
        ax.imshow(img)
    for ax,img,bxs in zip(axs[1],image_list[5:],poly_xy_list[5:]):
        for bx in bxs:
            cv2.polylines(img,[bx.reshape((-1,1,2)).astype(np.int32)],True,color,thickness=thickness)
        ax.imshow(img)
    plt.show()
        
