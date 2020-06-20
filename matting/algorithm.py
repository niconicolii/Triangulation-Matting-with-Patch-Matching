## CSC320 Winter 2020 
## Assignment 1
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

# import basic packages
import numpy as np
import scipy.linalg as sp
import cv2 as cv

# If you wish to import any additional modules
# or define other utility functions, 
# include them here

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################


#########################################

#
# The Matting Class
#
# This class contains all methods required for implementing 
# triangulation matting and image compositing. Description of
# the individual methods is given below.
#
# To run triangulation matting you must create an instance
# of this class. See function run() in file run.py for an
# example of how it is called
#
class Matting:
    #
    # The class constructor
    #
    # When called, it creates a private dictionary object that acts as a container
    # for all input and all output images of the triangulation matting and compositing 
    # algorithms. These images are initialized to None and populated/accessed by 
    # calling the the readImage(), writeImage(), useTriangulationResults() methods.
    # See function run() in run.py for examples of their usage.
    #
    def __init__(self):
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

    # Return a dictionary containing the input arguments of the
    # triangulation matting algorithm, along with a brief explanation
    # and a default filename (or None)
    # This dictionary is used to create the command-line arguments
    # required by the algorithm. See the parseArguments() function
    # run.py for examples of its usage
    def mattingInput(self): 
        return {
            'backA':{'msg':'Image filename for Background A Color','default':None},
            'backB':{'msg':'Image filename for Background B Color','default':None},
            'compA':{'msg':'Image filename for Composite A Color','default':None},
            'compB':{'msg':'Image filename for Composite B Color','default':None},
        }
    # Same as above, but for the output arguments
    def mattingOutput(self): 
        return {
            'colOut':{'msg':'Image filename for Object Color','default':['color.tif']},
            'alphaOut':{'msg':'Image filename for Object Alpha','default':['alpha.tif']}
        }
    def compositingInput(self):
        return {
            'colIn':{'msg':'Image filename for Object Color','default':None},
            'alphaIn':{'msg':'Image filename for Object Alpha','default':None},
            'backIn':{'msg':'Image filename for Background Color','default':None},
        }
    def compositingOutput(self):
        return {
            'compOut':{'msg':'Image filename for Composite Color','default':['comp.tif']},
        }
    
    # Copy the output of the triangulation matting algorithm (i.e., the 
    # object Color and object Alpha images) to the images holding the input
    # to the compositing algorithm. This way we can do compositing right after
    # triangulation matting without having to save the object Color and object
    # Alpha images to disk. This routine is NOT used for partA of the assignment.
    def useTriangulationResults(self):
        if (self._images['colOut'] is not None) and (self._images['alphaOut'] is not None):
            self._images['colIn'] = self._images['colOut'].copy()
            self._images['alphaIn'] = self._images['alphaOut'].copy()

    # If you wish to create additional methods for the 
    # Matting class, include them here

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    # check if all required images are read into memory
    #
    # Input arguments
    #     - keys:    list of images required given by key names
    # Return value
    #     - True if all required images are in memory, False if missing image
    #     - error message if missing image
    #     - list of images corresponding to given keys
    def _req_img_read(self, keys):
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
    #########################################
            
    # Use OpenCV to read an image from a file and copy its contents to the 
    # matting instance's private dictionary object. The key 
    # specifies the image variable and should be one of the
    # strings in lines 54-63. See run() in run.py for examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # leave the matting instance's dictionary entry unaffected and return
    # False, along with an error message
    def readImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
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
        #########################################
        return success, msg

    # Use OpenCV to write to a file an image that is contained in the 
    # instance's private dictionary. The key specifies the which image
    # should be written and should be one of the strings in lines 54-63. 
    # See run() in run.py for usage examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # return False, along with an error message
    def writeImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
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
        #########################################
        return success, msg

    # Method implementing the triangulation matting algorithm. The
    # method takes its inputs/outputs from the method's private dictionary 
    # ojbect. 
    def triangulationMatting(self):
        """
success, errorMessage = triangulationMatting(self)
        
        Perform triangulation matting. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        """

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
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
            # calculate the foreground object color
            #Fb = alpha * (c_ab + c_bb) / 2
            #Fg = alpha * (c_ag + c_bg) / 2
            #Fr = alpha * (c_ar + c_br) / 2
	    Fb = alpha * c_bb
            Fg = alpha * c_bg
            Fr = alpha * c_br
            # store color and alpha images to instance
            self._images['colOut'] = cv.merge((Fb, Fg, Fr))
            self._images['alphaOut'] = alpha  * 255
        #########################################

        return success, msg

        
    def createComposite(self):
        """
success, errorMessage = createComposite(self)
        
        Perform compositing. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
"""

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
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
        #########################################

        return success, msg


