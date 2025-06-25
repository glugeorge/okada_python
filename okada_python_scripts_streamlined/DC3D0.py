import numpy as np

from .calculate_constants import DCCON0, DCCON1

from .terms import UA0, UB0, UC0

def DC3D0(alpha,x,y,z,depth,dip,pot1,pot2,pot3,pot4):
    '''
    Inputs
    alpha: medium constant
    x,y,z: coordinates of observing point. 3D meshes defined along n_x, n_y, n_z
    depth: depth of reference point
    dip: dip-angle (degrees)
    pot1-pot4: strike-, dip-, tensile- and inflate-potency


    Outputs
    ux,uy,uz: displacements
    
    '''

    # initialize some arrays
    n_x, n_y, n_z = x.shape
    u = np.zeros((n_x, n_y, n_z,12))
    du = np.zeros((n_x, n_y, n_z,12))
    dua = np.zeros((n_x, n_y, n_z,12))
    dub = np.zeros((n_x, n_y, n_z,12))
    duc = np.zeros((n_x, n_y, n_z,12))

    # make sure z is negative
    if (z > 0).any() :
        print('z must be negative')
        return u
    
    # calculate medium constants
    c0 = DCCON0(alpha,dip)

    # real-source contributions
    d = depth + z
    c1 = DCCON1(x,y,d,c0)
    if (c1['r'] == 0).any():
        return u
    dua = UA0(x,y,d,pot1,pot2,pot3,pot4,c0,c1)
    u[...,:9] = u[...,:9] - dua[...,:9]
    u[...,9:] = u[...,9:] + dua[...,9:]

    # image-source contribution
    d = depth - z
    c1 = DCCON1(x,y,d,c0)
    dua = UA0(x,y,d,pot1,pot2,pot3,pot4,c0,c1)
    dub = UB0(x,y,d,z,pot1,pot2,pot3,pot4,c0,c1)
    duc = UC0(x,y,d,z,pot1,pot2,pot3,pot4,c0,c1)
    for i in range(12):
        du[...,i] = dua[...,i] + dub[...,i] + z*duc[...,i]
        if i >= 9:
            du[...,i] = du[...,i] + duc[...,i-9]
        u[...,i] = u[...,i]+du[...,i]
    return u