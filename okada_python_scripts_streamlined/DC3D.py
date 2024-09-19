import numpy as np

from .calculate_constants import DCCON0, DCCON2

from .terms import UA, UB, UC

def DC3D(alpha,x,y,z,depth,dip,al1,al2,aw1,aw2,disl1,disl2,disl3):
    '''
    Inputs
    alpha: medium constant
    x,y,z: coordinates of observing point. 3D meshes defined along n_x, n_y, n_z
    depth: depth of reference point
    dip: dip-angle (degrees)
    al1,al2: fault length range
    aw1,aw2: fault width range
    disl1, disl2, disl3: strike, dip, tensile dislocations

    Outputs
    ux,uy,uz: displacements
    
    '''
    # initialize some constants
    eps = 1e-6 # for when values near zero

    # initialize some arrays
    n_x, n_y, n_z = x.shape
    xi = np.zeros((n_x, n_y, n_z,2))
    et = np.zeros((n_x, n_y, n_z,2))
    kxi = np.zeros((n_x, n_y, n_z,2))
    ket = np.zeros((n_x, n_y, n_z,2))
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
    sd = c0['sd']
    cd = c0['cd']
    xi[...,0] = x - al1
    xi[...,1] = x - al2
    xi[np.abs(xi) < eps] = 0

    # real-source contributions
    d = depth + z
    p = y*cd + d*sd
    q = y*sd - d*cd
    et[...,0] = p-aw1
    et[...,1] = p-aw2
    q[np.abs(q) < eps] = 0
    et[np.abs(et) < eps] = 0

    # reject singular case
    ## on fault edge
    if ((q == 0).any() and (((xi[...,0]*xi[...,1] <= 0).any() and (et[...,0]*et[...,1]==0).any()) or ((et[...,0]*et[...,1]<=0).any() and (xi[...,0]*xi[...,1] == 0).any()))):
        print('Singular case on fault edge')
        return u
    
    ## on negative extension of fault edge
    # Compute r12, r21, r22 for all elements
    r12 = np.sqrt(xi[..., 0]**2 + et[..., 1]**2 + q*q)
    r21 = np.sqrt(xi[..., 1]**2 + et[..., 0]**2 + q*q)
    r22 = np.sqrt(xi[..., 1]**2 + et[..., 1]**2 + q*q)

    # Apply the conditions element-wise and update kxi and ket accordingly
    kxi[..., 0] = np.where((xi[..., 0] < 0) & (r21 + xi[..., 1] < eps), 1, kxi[..., 0])
    kxi[..., 1] = np.where((xi[..., 0] < 0) & (r22 + xi[..., 1] < eps), 1, kxi[..., 1])

    ket[..., 0] = np.where((et[..., 0] < 0) & (r12 + et[..., 1] < eps), 1, ket[..., 0])
    ket[..., 1] = np.where((et[..., 0] < 0) & (r22 + et[..., 1] < eps), 1, ket[..., 1])

    for k in range(2):
        for j in range(2):
            # CALL DCCON2(XI(J),ET(K),Q,SD,CD,KXI(K),KET(J))
            c2 = DCCON2(xi[...,j],et[...,k],q,sd,cd,kxi[...,k],ket[...,j])

            # CALL UA(XI(J),ET(K),Q,DD1,DD2,DD3,DUA)
            # in fortran, dd1,dd2,dd3 corespond to disl1,disl2,disl3
            dua = UA(xi[...,j],et[...,k],q,disl1,disl2,disl3,c0,c2)
            for i in range(0,10,3):
                du[...,i] = -dua[...,i]
                du[...,i+1] = -dua[...,i+1]*cd + dua[...,i+2]*sd
                du[...,i+2] = -dua[...,i+1]*sd - dua[...,i+2]*cd
                if i == 9:
                    du[...,i] = -du[...,i]
                    du[...,i+1] = -du[...,i+1]
                    du[...,i+2] = -du[...,i+2]
            if j+k == 1:
                u = u - du
            else:
                u = u + du

    # image-source contribution
    d = depth - z
    p = y*cd + d*sd
    q = y*sd - d*cd
    et[...,0] = p-aw1
    et[...,1] = p-aw2
    q[np.abs(q) < eps] = 0
    et[np.abs(et) < eps] = 0

    # reject singular case
    ## on fault edge
    if ((q == 0).any() and (((xi[...,0]*xi[...,1] <= 0).any() and (et[...,0]*et[...,1]==0).any()) or ((et[...,0]*et[...,1]<=0).any() and (xi[...,0]*xi[...,1] == 0).any()))):
        print('Singular case on fault edge')
        return u
    ## on negative extension of fault edge
    kxi = np.zeros((n_x, n_y, n_z,2))
    ket = np.zeros((n_x, n_y, n_z,2))
    # Compute r12, r21, r22 for all elements
    r12 = np.sqrt(xi[..., 0]**2 + et[..., 1]**2 + q*q)
    r21 = np.sqrt(xi[..., 1]**2 + et[..., 0]**2 + q*q)
    r22 = np.sqrt(xi[..., 1]**2 + et[..., 1]**2 + q*q)

    # Apply the conditions element-wise and update kxi and ket accordingly
    kxi[..., 0] = np.where((xi[..., 0] < 0) & (r21 + xi[..., 1] < eps), 1, kxi[..., 0])
    kxi[..., 1] = np.where((xi[..., 0] < 0) & (r22 + xi[..., 1] < eps), 1, kxi[..., 1])

    ket[..., 0] = np.where((et[..., 0] < 0) & (r12 + et[..., 1] < eps), 1, ket[..., 0])
    ket[..., 1] = np.where((et[..., 0] < 0) & (r22 + et[..., 1] < eps), 1, ket[..., 1])

    for k in range(2):
        for j in range(2):
            c2 = DCCON2(xi[...,j],et[...,k],q,sd,cd,kxi[...,k],ket[...,j])
            dua = UA(xi[...,j],et[...,k],q,disl1,disl2,disl3,c0,c2)
            dub = UB(xi[...,j],et[...,k],q,disl1,disl2,disl3,c0,c2)
            duc = UC(xi[...,j],et[...,k],q,z,disl1,disl2,disl3,c0,c2)

            for i in range(0,10,3):
                du[...,i] = dua[...,i] + dub[...,i] + z*duc[...,i]
                du[...,i+1] = (dua[...,i+1]+dub[...,i+1]+z*duc[...,i+1])*cd  - (dua[...,i+2]+dub[...,i+2]+z*duc[...,i+2])*sd
                du[...,i+2] = (dua[...,i+1]+dub[...,i+1]-z*duc[...,i+1])*sd  + (dua[...,i+2]+dub[...,i+2]-z*duc[...,i+2])*cd
                if i == 9:
                    du[...,9] = du[...,9] + duc[...,0]
                    du[...,10] = du[...,10] + duc[...,1]*cd - duc[...,2]*sd
                    du[...,11] = du[...,11] - duc[...,1]*sd - duc[...,2]*cd
            if j+k == 1:
                u = u - du
            else:
                u = u + du
    #ux = u[0]
    #uy = u[1]
    #uz = u[2]
    #uxx = u[3]
    #uyx = u[4]
    #uzx = u[5]
    #uxy = u[6]
    #uyy = u[7]
    #uzy = u[8]
    #uxz = u[9]
    #uyz = u[10]
    #uzz = u[11]

    return u