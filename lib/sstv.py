from collections import Iterable,defaultdict
import numpy as np
import cv2

def np_apply_matrix_to_pts(M,pts):
    """
    Apply transform matrix to points
    Args:
        M: (axis1,axis2) matrix
        pts: (k1,k2,...,kn,axis) points coordinate
    Return:
        (k1,k2,...,kn,axis1) points coordinate
    """
    len2 = False
    if(isinstance(pts,list) or pts.dtype==np.object):
        ans = [np_apply_matrix_to_pts(M,o) for o in pts]
        return np.array(ans)
    if(pts.shape[-1]<M.shape[-1]):
        # need homogeneous, (ax0,ax1...)->(ax0,ax1,...,1)
        len2 = True
        pd = [(0,0) for i in range(len(pts.shape)-1)]
        pd.append((0,1))
        pts = np.pad(pts,pd,constant_values=1)
    if(len(pts.shape)>1):
        pts = np.moveaxis(pts,-1,-2)
    ret = np.dot(M,pts)
    if(len2):
        ret = ret[:-1]
    ret = np.moveaxis(ret,0,-1)
    return ret

def cv_gen_trajectory(image:np.ndarray,total_step:int,
    poly_xy=None,fluctuation:float=0.1,**args
    ):
    """
    Generate image sequence from single trajectory.
    Args:
        image: input image, must be type of np.uint8
        total_step: number of key frame in trajectory
        poly_xy (optional): polygon represented as xy coordinates with shape ((N),k,2)
            N: number of polygon; k: number of vertex in single polygon 
        fluctuation: set fluctuation higher than 0 to enable fluctuation when dividing trajectory
            range in [-fluctuation,+fluctuation]+1.0

        ==trajectory args, optional==
        rotate: float of final rotation angle
        shift: (y,x) shift or single float 
        scale: (y,x) scale or single float
        return_states: bool, set True to return parameters in each step
        blur: bool, set True to enable blur kernel
            blur_rate: float, blur rate, default is 0.3
            blur_stepi: list of int, value range in [1,total_step-1], 
                designate the index of blured image
            blur_ksize: int, kernel size, default is 10
            blur_intensity: float, intensity, default is 0.1
            blur_return_stepi: bool, if True return blur_stepi 
            blur_motion: bool, set True to enable motion blur calculate from coordinate
            [CAUTION] enable blur_motion will cause huge calculation

    Return:
        image_list: list of images, each item corresponds to a frame
        poly_xy_list: return list of polygons if polygon is provided and None if polygon is not provided, 
            each item corresponds to a frame
        Ms: list of affine matrix, each item corresponds to the mapping of original image to current
        (step_state_dict): if return_states is True, return a dictionary of trajectory args in each step
            e.g. step_state_dict = {'rotate':[0,10,20...],'shiftx':[0,-4,-8...]...}
        (blur_stepi): if blur_return_stepi is True, return a list to mark blured images 
    """
    org_image_size=image.shape[:-1]
    Ms = [np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)]
    image_list = [image]
    rotate = args['rotate'] if('rotate' in args)else 0.0
    shift = args['shift'] if('shift' in args)else (0,0)
    scale = args['scale'] if('scale' in args)else (0,0)
    fluctuation = min(0.3,abs(fluctuation))
    poly_xy_list = None
    if(not isinstance(shift,Iterable)):
        shift = (shift,shift)
    if(not isinstance(scale,Iterable)):
        scale = (scale,scale)
    if(poly_xy is not None):
        poly_xy_list = [np.expand_dims(poly_xy,0) if(len(poly_xy.shape)==2)else poly_xy]
    scale_det = (scale[0]-1,scale[1]-1)
    seed = np.random

    if('blur' in args and args['blur']):
        rate = min(0.7,args['blur_rate']) if('blur_rate' in args) else 0.3
        if('blur_stepi' in args):
            blur_stepi = [stepi-1 for stepi in args['blur_stepi'] if(stepi<total_step)]
        else:
            blur_stepi = [stepi for stepi in range(total_step-2) if(seed.random()>rate)]
    else:
        blur_stepi = []

    state_dict = {
        'rotate':[0],
        'shiftx':[0],
        'shifty':[0],
        'scalex':[1],
        'scaley':[1],
    }
    # rotate,scale_xy,shift_xy
    last = [0,1,1,0,0]
    for stepi in range(total_step-1):
        Mlst = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        cur = []
        if('rotate' in args):
            f = (stepi+1)/(total_step-1)
            if(fluctuation>0 and (stepi+1)!=(total_step-1)):
                f*=((seed.random()-0.5)*2*fluctuation+1)
            Mr = cv2.getRotationMatrix2D(((org_image_size[1]-1)/2,(org_image_size[0]-1)/2), rotate*f, 1.0)
            Mr = np.concatenate((Mr,np.array([[0,0,1]],dtype=Mr.dtype)),0)
            Mlst = np.dot(Mr,Mlst)
            state_dict['rotate'].append(rotate*f)
            cur.append(rotate*f)
        else:
            state_dict['rotate'].append(0)
            cur.append(0)

        if('scale' in args):
            fx = (stepi+1)/(total_step-1)
            fy = fx
            if(fluctuation>0 and (stepi+1)!=(total_step-1)):
                fx*=((seed.random()-0.5)*2*fluctuation+1)
                fy*=((seed.random()-0.5)*2*fluctuation+1)
            Msc = np.array([[scale_det[1]*fx+1,0,0],[0,scale_det[0]*fy+1,0],[0,0,1]],dtype=Mlst.dtype)
            Mlst = np.dot(Msc,Mlst)
            state_dict['scalex'].append(scale_det[1]*fx)
            state_dict['scaley'].append(scale_det[0]*fy)
            cur+=[scale_det[1]*fx+1,scale_det[0]*fy+1]
        else:
            state_dict['scalex'].append(1)
            state_dict['scaley'].append(1)
            cur+=[1,1]

        if('shift' in args):
            fx = (stepi+1)/(total_step-1)
            fy = fx
            if(fluctuation>0 and (stepi+1)!=(total_step-1)):
                fx*=((seed.random()-0.5)*2*fluctuation+1)
                fy*=((seed.random()-0.5)*2*fluctuation+1)
            Mt = np.array([[1,0,shift[1]*fx],[0,1,shift[0]*fy],[0,0,1]],dtype=Mlst.dtype)
            Mlst = np.dot(Mt,Mlst)
            state_dict['shiftx'].append(shift[1]*fx)
            state_dict['shifty'].append(shift[0]*fy)
            cur+=[shift[1]*fx,shift[0]*fy]
        else:
            state_dict['shiftx'].append(0)
            state_dict['shifty'].append(0)
            cur+=[0,0]

        Ms.append(Mlst)
        img = cv2.warpAffine(image_list[0], Mlst[:-1], org_image_size[::-1])
        # ensure last image is no-blur
        if(stepi in blur_stepi and 'blur' in args and args['blur']):
            ksize = int(args['blur_ksize']) if('blur_ksize' in args)else 11
            intensity = float(args['blur_intensity']) if('blur_intensity' in args)else 0.0
            intensity = max(0,min(0.3,intensity))

            # CAUTION: this custom function (enable 'blur_motion') will be EXTREMELY slow
            # we can't directly use the convolution function in cv2 because each point 
            # has different kernels, but there is no convolution function for pixel level 
            # kernel convolution hence we provide an simple implementation.
            if('blur_motion' in args and args['blur_motion']):
                # original coordinate -> M dot -> new coordinate
                # difference === O[t]-O[t-1] so new coordinate is difference
                blur_M = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
                blur_Mr = cv2.getRotationMatrix2D(((org_image_size[1]-1)/2,(org_image_size[0]-1)/2), cur[0]-last[0], 1.0)
                blur_Mr = np.concatenate((blur_Mr,np.array([[0,0,1]],dtype=blur_Mr.dtype)),0)
                blur_M = np.dot(blur_Mr,blur_M)
                blur_Msc = np.array([[cur[1]/last[1],0,0],[0,cur[2]/last[2],0],[0,0,1]],dtype=blur_M.dtype)
                blur_M = np.dot(blur_Msc,blur_M)
                blur_Mt = np.array([[1,0,cur[3]-last[3]],[0,1,cur[4]-last[4]],[0,0,1]],dtype=blur_M.dtype)
                blur_M = np.dot(blur_Mt,blur_M)
                cox,coy = np.arange(org_image_size[1]),np.arange(org_image_size[0])
                cox,coy = np.meshgrid(cox,coy)
                # shape (H,W,3) with (x,y,1)
                co_org = np.stack([cox,coy,np.ones(cox.shape,dtype=cox.dtype)],-1)
                co_new = np_apply_matrix_to_pts(blur_M,co_org)

                # draw line on each point and normalize, slow
                def kernel_helper(dx,dy,ksize):
                    # draw line function
                    sp = np.array((0,0))
                    if(dy!=0):
                        k_shift = dx/dy
                        if(k_shift>0.05):
                            ep = np.array((ksize,k_shift*ksize))
                        elif(k_shift<-0.05):
                            sp = np.array((0,ksize))
                            ep = np.array((ksize,k_shift*ksize+ksize))
                        else:
                            sp = np.array((0,ksize//2))
                            ep = np.array((ksize,ksize//2))
                    else:
                        ep = np.array((0,ksize))
                    
                    sp = sp.astype(np.int32)
                    ep = ep.astype(np.int32)
                    kernel = np.zeros((ksize,ksize),dtype=np.uint8)
                    kernel = cv2.line(kernel,(sp[0],sp[1]),(ep[0],ep[1]),color=1,thickness=1)
                    matrix_k = np.sum(kernel>0)
                    if(matrix_k>1):
                        kernel = np.divide(kernel,matrix_k)
                    else:
                        if(ksize%2==1):
                            kernel[ksize//2,ksize//2]=1
                        else:
                            kernel[ksize//2-1,ksize//2-1]=1
                    return kernel
                f_helper = lambda o:kernel_helper(o[0],o[1],ksize)
                # H,W,ksize,ksize array
                kernel_map = np.apply_along_axis(f_helper,-1,co_new.reshape(-1,3)).reshape(co_new.shape[0],co_new.shape[1],ksize,ksize)
                new_img = np.zeros(img.shape,dtype = image.dtype)
                # apply convolution
                hksize = ksize//2
                lksize = hksize - (1 if(ksize%2==0)else 0)
                for di in range(img.shape[0]):
                    for dj in range(img.shape[1]):
                        spi,spj = max(0,di - lksize),max(0,dj-lksize)
                        patch = img[
                            spi:min(img.shape[0]-1,di+hksize),
                            spj:min(img.shape[1]-1,dj+hksize),
                            ]
                        kspi,kspj = lksize-(di-spi),lksize-(dj-spj)
                        kr = kernel_map[di,dj,kspi:kspi+patch.shape[0],kspj:kspj+patch.shape[1]]
                        new_img[di,dj] = np.sum(patch*np.stack([kr,kr,kr],-1),axis=(0,1))
                img = new_img

            elif('shift' in args):
                kr_shift = np.zeros((ksize,ksize))
                fdet = 1/(total_step-1)
                # start point in X,Y
                sp = np.array((0,0))
                # find end point
                if(shift[1]!=0):
                    k_shift = shift[0]/shift[1]
                    k_shift += k_shift*(seed.random()-0.5)*2*intensity
                    if(k_shift>0.05):
                        ep = np.array((ksize,k_shift*ksize))
                    elif(k_shift<-0.05):
                        sp = np.array((0,ksize))
                        ep = np.array((ksize,k_shift*ksize+ksize))
                    else:
                        sp = np.array((0,ksize//2))
                        ep = np.array((ksize,ksize//2))
                else:
                    ep = np.array((0,ksize))
                sp = sp.astype(np.int32)
                ep = ep.astype(np.int32)
                kr_shift = cv2.line(kr_shift,(sp[0],sp[1]),(ep[0],ep[1]),color=1,thickness=1)
                matrix_k = np.sum(kr_shift>0)
                if(matrix_k>1):
                    kr_shift /= matrix_k
                    img = cv2.filter2D(img,-1,kr_shift)
            if('gaussian' in args):
                blur = cv2.GaussianBlur(img,(ksize,ksize),0)
        image_list.append(img)
        if(poly_xy is not None and len(poly_xy)>0):
            poly_xy_list.append(np_apply_matrix_to_pts(Mlst,poly_xy))
        last = cur

    ans = [image_list,poly_xy_list,Ms]
    if('return_states' in args and args['return_states']):
        ans.append(state_dict)
    if('blur' in args and 'blur_return_stepi' in args and args['blur_return_stepi']):
        blur_stepi = [o+1 for o in blur_stepi]
        ans.append(blur_stepi)

    return ans