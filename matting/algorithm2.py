# import basic packages
import numpy as np
import scipy.linalg as sp
import cv2 as cv
from copy import deepcopy as copy

class Matting:
    
    def __init__(self):
        """ 
        Contruct a dictionary for all input and output images for applying
        Triangulation Matting and Composition.
        """        
        self._images = { 
            'backA': None, 
            'backB': None, 
            'compA': None, 
            'compB': None, 
            'colOut': None,
            'alphaOut': None, 
            'backIn': None, 
            'colIn': None, 
            'alphaIn': None, 
            'compOut': None, 
        }

    def mattingInput(self): 
        """
        Dictionary with explainations of all input files for Triangular Matting.
        """
        return {
            'backA':{'msg':'Image filename for Background A Color','default':None},
            'backB':{'msg':'Image filename for Background B Color','default':None},
            'compA':{'msg':'Image filename for Composite A Color','default':None},
            'compB':{'msg':'Image filename for Composite B Color','default':None},
        }

    def mattingOutput(self): 
        """
        Dictionary with explanations of all output files for Triangular Matting.
        """
        return {
            'colOut':{'msg':'Image filename for Object Color','default':['color.tif']},
            'alphaOut':{'msg':'Image filename for Object Alpha','default':['alpha.tif']}
        }
    
    def compositingInput(self):
        """
        Dictionary with explainations of all input files for Composition.
        """        
        return {
            'colIn':{'msg':'Image filename for Object Color','default':None},
            'alphaIn':{'msg':'Image filename for Object Alpha','default':None},
            'backIn':{'msg':'Image filename for Background Color','default':None},
        }
    
    def compositingOutput(self):
        """
        Dictionary with explainations of all output files for Composition.
        """        
        return {
            'compOut':{'msg':'Image filename for Composite Color','default':['comp.tif']},
        }
    
    def useTriangulationResults(self):
        """
        Save output of Triangulation Matting to corresponding varaibles.
        """
        if (self._images['colOut'] is not None) and (self._images['alphaOut'] is not None):
            self._images['colIn'] = self._images['colOut'].copy()
            self._images['alphaIn'] = self._images['alphaOut'].copy()

    
    
    def _req_img_read(self, keys):
        """
        Check if all required images are read into memory
        
        Parameters:
            keys    list of images required given by key names
        Return:
            success  True if all required images are read, otherwise False
            msg      error message if missing image
            imgs     list of images corresponding to given keys
        """
        success = True
        msg = 'Placeholder'
        imgs = []       # list of images corresponding to given keys
        miss_img = []   # list of key names of missing images
        for k in keys:
            img = self._images[k]
            imgs.append(img)
            if img is None:
                miss_img.append(k)
        if len(miss_img) != 0:
            msg = "Missing image(s) for {}".format(', '.join(miss_img))
            success = False
        return success, msg, imgs
            
    # Use OpenCV to read an image from a file and copy its contents to the 
    # matting instance's private dictionary object. The key 
    # specifies the image variable and should be one of the
    # strings in lines 54-63. See run() in run.py for examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # leave the matting instance's dictionary entry unaffected and return
    # False, along with an error message
    def readImage(self, fileName, key):
        """
        Read image by given file name with openCVand  and store it to the
        dictionary with given key name.
        
        Parameters:
            fileName    string containing file name of image
            key         string containing corresponding key name in the dictionary
        Return:
            success     True if image read into memory, otherwise False
            msg         error message to print out when fail to read image
        """
        success = False
        msg = 'Placeholder'
        
        # try read image using openCV, read image in grayscale if key is 'alphaIn'
        if key == 'alphaIn':
            rd_img = cv.imread(fileName, 0)
        else:
            rd_img = cv.imread(fileName)
            
        # if succeed reading image, store to the matting instance's dictionary
        # entry
        if not rd_img is None:
            self._images[key] = rd_img.astype('float')
            success = True
        # if failed reading image, return False with error message
        else:
            msg = 'Failed to read file {} for image {}'.format(fileName, key)
            
        return success, msg
    
    def writeImage(self, fileName, key):
        """
        Write computed image to file.
        
        Parameters:
            fileName    string containing file name of image to be written
            key         string containing corresponding key name in the dictionary
        Return:
            success     True if image read into memory, otherwise False
            msg         error message to print out when fail to read image
        """        
        success = False
        msg = 'Placeholder'

        # writing must fail if there is no image by given key ready to be output 
        wt_img = self._images[key]
        
        if wt_img is None:
            msg = "Nothing in image {} to be written".format(key)
        else:
            # try to write output image to file
            written = cv.imwrite(fileName, wt_img)
            
            # if failed to write image, return False with error msg
            if not written:
                msg = "Failed to write image {} to {}.".format(key, fileName)
            else:
                success = True
        return success, msg

    def triangulationMatting(self):
        """        
        Perform triangulation matting with patch match evaluations. Inputs are 
        taken from Matting's dictionary.
        
        Return:
            success     False if can't find all inputs
            msg         error message to print out when fail to perform matting
        """
        success = False
        msg = 'Placeholder'
        
        # number of equations
        eq_num = 6
        
        # matting fails if missing image
        keys = ['compA', 'compB', 'backA', 'backB']
        success, msg, imgs = self._req_img_read(keys)
        
        if success:
            # split images into B,G,R
            c_ab, c_ag, c_ar = cv.split(imgs[0])
            c_bb, c_bg, c_br = cv.split(imgs[1])
            b_ab, b_ag, b_ar = cv.split(imgs[2])
            b_bb, b_bg, b_br = cv.split(imgs[3])
            
            # calculate alpha using equation from Smith and Blinn's 
            # Blue Screen Matting paper
            alpha =  1 - ((c_ab-c_bb)*(b_ab-b_bb) + (c_ag-c_bg)*(b_ag-b_bg) +\
                         (c_ar-c_br)*(b_ar-b_br)) / ((b_ab-b_bb)**2 +\
                         (b_ag-b_bg)**2 + (b_ar-b_br)**2)
            h_radius = 167
            w_radius = 215
            print "236"
            max_coord = find_limited_area_center_coord(alpha, h_radius, w_radius)
            print "238"
            ## extent h and w radius 
            #h_radius = 172
            #w_radius = 192
            a_l = get_limited_area(imgs[0], h_radius, w_radius, max_coord)
            b_l = get_limited_area(imgs[1], h_radius, w_radius, max_coord)
            print "241"
            alpha_l =  get_limited_area(alpha, h_radius, w_radius, max_coord)
            print "243"
            alpha_l_a = copy(alpha_l)
            alpha_l_b = copy(alpha_l)
            
            for i in range(2):
                alpha_l_a = update_alpha_by_patchmatch(alpha_l_a, alpha_l_b, 
                                                       a_l, b_l, h_radius, w_radius)
                alpha_l_b = update_alpha_by_patchmatch(alpha_l_b, alpha_l_a,
                                                       b_l, a_l, h_radius, w_radius)
            
            y,x = max_coord
            alpha = alpha * 0
            alpha[y-h_radius:y+h_radius+1, x-w_radius:x+w_radius+1] = alpha_l_a
            # calculate the foreground object color
            Fb = alpha * c_ab
            Fg = alpha * c_ag
            Fr = alpha * c_ar
            #Fb = alpha * c_bb
            #Fg = alpha * c_bg
            #Fr = alpha * c_br            
            # store color and alpha images to instance
            self._images['colOut'] = cv.merge((Fb, Fg, Fr))
            self._images['alphaOut'] = alpha  * 255

        return success, msg

        
    def createComposite(self):
        """
        Perform copositing with inputs in the matting dictionary.
        
        Return:
            success     False if can't find required inputs
            msg         error message to print out when fail to perform matting
        """
        success = False
        msg = 'Placeholder'
        
        # compositing fails if missing image
        keys = ['backIn', 'colIn', 'alphaIn']
        success, msg, imgs = self._req_img_read(keys)
        
        if success:
            # split images into B,G,R
            Bb, Bg, Br = cv.split(imgs[0])
            Fb, Fg, Fr = cv.split(imgs[1])
            
            # make alpha image's into numbers 0 - 1
            alpha = imgs[2] / 255
            
            # calculate Composite color
            Cb = alpha * Fb + (1 - alpha) * Bb
            Cg = alpha * Fg + (1 - alpha) * Bg
            Cr = alpha * Fr + (1 - alpha) * Br
            self._images['compOut'] = cv.merge((Cb, Cg, Cr))

        return success, msg

def find_limited_area_center_coord(alpha, h_radius, w_radius):
    """
    Find the center of an area with given radius with most number of pixels
    that are part of the front ground object.
    
    Parameters:
        alpha        alpha channel image
        h_radius     radius of height of area to find
        w_radius     radius of width of area to find
    
    Return:
        max_coord    coordinate of center of found area
    """
    (N,M) = alpha.shape
    h = h_radius * 2 + 1
    w = w_radius * 2 + 1
    max_coord = None
    max_sum = 0
    for r in range(h_radius, N-h_radius):
        for c in range(w_radius, M-w_radius):
            rc_sum = np.sum(alpha[r-h_radius:r+h_radius, c-w_radius:c+w_radius])
            if rc_sum > max_sum:
                max_sum = rc_sum
                max_coord = np.array([r,c])
    return max_coord
    
def get_limited_area(im, h_r, w_r, center):
    """
    Get the area by its center and radiuses
    
    Parameter:
        im     representation of the whole image
        h_r    radius of height of area
        w_r    radius of width of area
        center center coordinate of area
    Return:
        area found
    """
    y,x = center
    return im[y-h_r:y+h_r+1, x-w_r:x+w_r+1]


def make_patch_matrix(im, patch_size):
    """
    Create the matrix where each element is a patch of surrounding pixels with
    pixel on element is at center.
    
    Parameter:
        im            representation of the whole image
        patch_size    the size of neighbouring pixels to consider
    Return
        patch_matrix
    """
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))


def get_distance(m0, m1, is_list=True):
    """
    Helper funtion for calculating the distance of patches between two list of 
    patches that are NxCxP**2, or a pair of patches that are CxP**2, is_list
    parameter indicates that m0 and m1 are list
    """
    # if it is a matrix, then 
    if is_list:
        (N,C,P) = m0.shape
        # reshape two matrices for calculations
        m0 = np.nan_to_num(m0.reshape((N, C*P)))
        m1 = m1.reshape((N, C*P))
        # get the differences between each corresponding values, make all nan
        # elements to have the biggest difference possible
        diff = np.absolute(m0 - m1)
        nan_i = np.where(np.isnan(diff))
        diff[nan_i] = np.inf
        # find the mean of sqaure distances 
        best_D = np.sum(diff ** 2, axis=1) / (C*P)
    # if it is a patch centered at a single pixel, then:
    else:
        (C,P) = m0.shape
        # reshape two matrices for calculations
        m0 = m0.reshape(C*P)
        m1 = m1.reshape(C*P)
        # get the differences between each corresponding values, make all nan
        # elements to 0 to avoid making result all nans
        diff = np.nan_to_num(np.absolute(m0 - m1))
        # get dot product
        best_D = diff.dot(diff)
    
    return best_D

def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  it_num, best_D,
                                  alpha_list
                                ):
    new_f = f.copy()
    # number of rows and columns, and pathces size in this set of photo
    (N,M,C,P) = source_patches.shape
    R = int(P ** 0.5 // 2)
    # if the best_D is not initialized, create best_D
    if best_D is None:
        # create a matrix that patches in target are in order corresponding to
        # patches in source by NNF
        indices = make_coordinates_matrix((N, M))
        # clip to avoid out of bound
        nnf_row = np.clip((f + indices)[:,:,0], 0, N-1).astype(int)
        nnf_col = np.clip((f + indices)[:,:,1], 0, M-1).astype(int)
        new_f = np.dstack((nnf_row - indices[:,:,0], nnf_col - indices[:,:,1]))
        target = target_patches[nnf_row, nnf_col]
        best_D = get_distance(source_patches.reshape(N*M,C,P), 
                              target.reshape(N*M,C,P)).reshape(N,M)
    for r in range(N):
        for c in range(M):
            ### Propagation ###
            # set addition number for neighbour coordinates which is different in odd
            # iterations and even iterations
            it = 1
            if it_num // 2 != 0:
                it = -1
            cands = np.array([new_f[r,c]])
            # find displacement candidate from neighbour pixel on above or below
            if not (r+it < 0 or r+it >= N):
                cands = np.append(cands, [new_f[r+it,c]], axis=0)
            # find displacement candidate from neighbour pixel on above or below
            if not (c+it < 0 or c+it >= M):
                cands = np.append(cands, [new_f[r,c+it]], axis=0)
            cands_num = cands.shape[0]
            if cands_num != 1:
                cands_d = np.empty(cands_num)
                cands_d[0] = best_D[r,c]
                rows = np.clip(cands[1:,0] + r, 0, N-1).astype(int)
                cols = np.clip(cands[1:,1] + c, 0, M-1).astype(int)
                cands[1:] = np.dstack((rows - r, cols - c))[0]
                target_p = target_patches[rows, cols]
                source_p = np.tile(source_patches[r,c],
                                   (cands_num-1,1)).reshape(cands_num-1,C,P)
                cands_d[1:] = get_distance(source_p, target_p)
                min_i = np.argmin(cands_d)
                if min_i != 0:
                    new_f[r,c] = cands[min_i]
                    best_D[r,c] = cands_d[min_i]

            # create the list of alpha values if not yet defined
            if alpha_list is None:
                alpha_list = []
                exp = 1
                while (w * (alpha ** exp) >= 1):
                    alpha_list.append(alpha ** exp)
                    exp = exp + 1
                alpha_list = np.array(alpha_list)
            # number of alpha values
            n = alpha_list.shape[0]
            
            # n pairs of random coordinates using uniform random numbers in [-1,1]
            rand = 2 * np.random.random_sample((n,2)) - 1
            u = (rand.T * alpha_list.T).T * w + new_f[r,c]
            rows = np.clip(u[:,0] + r, 0, N-1).astype(int)
            cols = np.clip(u[:,1] + c, 0, M-1).astype(int)
            u = np.dstack((rows - r, cols - c))[0]
            target_r = target_patches[rows, cols]
            source_r = np.tile(source_patches[r,c], (n,1)).reshape(n,C,P)
            u_d = get_distance(source_r, target_r)
            min_i = np.argmin(u_d)
            if u_d[min_i] < best_D[r,c]:
                new_f[r,c] = u[min_i]
                best_D[r,c] = u_d[min_i]

    return new_f, best_D, alpha_list


def update_alpha_by_patchmatch(alpha_s, alpha_t, 
                               source, target, 
                               h_radius, w_radius):
    front_s = (source.T * alpha_s.T).T
    front_t = (target.T * alpha_t.T).T
    # initialize variables for patch match
    f = np.zeros((h_radius*2+1,w_radius*2+1,2))
    best_D = None
    alpha_list = None
    source = make_patch_matrix(front_s, 3)
    target = make_patch_matrix(front_t, 3)
    for i in range(5):
        print "Running", i, "th iteration for patchmaching"
        f, best_D, alpha_list = propagation_and_random_search(source,
                                                              target,
                                                              f, 0.5, w_radius,
                                                              i, best_D,
                                                              alpha_list)
    # we don't want to change the alpha value to change if the match is exactly
    # the same pixel
    for r in range(best_D.shape[0]):
        for c in range(best_D.shape[1]):
            y,x = f[r,c]
            if y <= 2 and x <= 2:
                best_D[r,c] = np.inf
    (g_rows, g_cols) = np.where(best_D <= 1000)            
    alpha_s[g_rows, g_cols] = np.clip(alpha_s[g_rows, g_cols] / 0.8, 0, 1)
    
    #(inf_rows, inf_cols) = np.where(best_D == np.inf)
    #best_D[inf_rows, inf_cols] = 0
    (b_rows, b_cols) = np.where(best_D >= 20000)
    alpha_s[b_rows, b_cols] = np.clip(alpha_s[b_rows, b_cols] * 0.5, 0, 1)
    
    return alpha_s