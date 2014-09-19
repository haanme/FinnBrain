'''
Created on Jun 26, 2014

@author: eija
'''

if __name__ == '__main__':
    pass

import numpy as np
import sys
import nrrd
import StringIO

#
# Calcualte FA maps
#
def calculate_FA(dwifile, bvecfile):
    import nibabel as nib
    from nibabel import linalg as al

    # Load data
    dwidata = nib.load(dwifile).get_data()

    bvecs = np.loadtxt(bvecfile)
    