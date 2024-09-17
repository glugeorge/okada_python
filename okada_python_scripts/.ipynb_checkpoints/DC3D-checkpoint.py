import numpy as np

from .calculate_constants import DCCON0, DCCON2

from .terms import UA, UB, UC

def DC3D(alpha,x,y,z,depth,dip,al1,al2,aw1,aw2,disl1,disl2,disl3):
    '''
    Inputs
    alpha: medium constant
    x,y,z: coordinate of observing point
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

    # make sure z is negative
    if z > 0:
        print('z must be negative')
        return 0,0,0,0,0,0,0,0,0,0,0,0

    # initialize some arrays
    xi = np.zeros(2)
    et = np.zeros(2)
    kxi = np.zeros(2)
    ket = np.zeros(2)
    u = np.zeros(12)
    du = np.zeros(12)
    dua = np.zeros(12)
    dub = np.zeros(12)
    duc = np.zeros(12)
    
    # calculate medium constants
    c0 = DCCON0(alpha,dip)
    sd = c0['sd']
    cd = c0['cd']
    xi[0] = x - al1
    xi[1] = x - al2
    if abs(xi[0]) < eps:
        xi[0] = 0
    if abs(xi[1]) < eps:
        xi[1] = 0

    # real-source contributions
    d = depth + z
    p = y*cd + d*sd
    q = y*sd - d*cd
    et[0] = p-aw1
    et[1] = p-aw2
    if abs(q) < eps:
        q = 0
    if abs(et[0]) < eps:
        et[0] = 0
    if abs(et[1]) < eps:
        et[1] = 0

    # reject singular case
    ## on fault edge
    if (q == 0 and ((xi[0]*xi[1] <= 0 and et[0]*et[1]==0) or (et[0]*et[1]<=0 and xi[0]*xi[1] == 0))):
        print('Singular case on fault edge')
        return 0,0,0,0,0,0,0,0,0,0,0,0
    ## on negative extension of fault edge
    kxi[0] = 0
    kxi[1] = 0
    ket[0] = 0
    ket[1] = 0
    r12 = np.sqrt(xi[0]*xi[0] + et[1]*et[1] + q*q)
    r21 = np.sqrt(xi[1]*xi[1] + et[0]*et[0] + q*q)
    r22 = np.sqrt(xi[1]*xi[1] + et[1]*et[1] + q*q)
    if (xi[0]<0 and r21+xi[1] < eps):
        kxi[0] = 1
    if (xi[0]<0 and r22+xi[1] < eps):
        kxi[1] = 1  
    if (et[0]<0 and r12+et[1] < eps):
        ket[0] = 1
    if (et[0]<0 and r22+et[1] < eps):
        ket[1] = 1

    for k in range(2):
        for j in range(2):
            # CALL DCCON2(XI(J),ET(K),Q,SD,CD,KXI(K),KET(J))
            c2 = DCCON2(xi[j],et[k],q,sd,cd,kxi[k],ket[j])

            # CALL UA(XI(J),ET(K),Q,DD1,DD2,DD3,DUA)
            # in fortran, dd1,dd2,dd3 corespond to disl1,disl2,disl3
            dua = UA(xi[j],et[k],q,disl1,disl2,disl3,c0,c2)
            for i in range(0,10,3):
                du[i] = -dua[i]
                du[i+1] = -dua[i+1]*cd + dua[i+2]*sd
                du[i+2] = -dua[i+1]*sd - dua[i+2]*cd
                if i == 9:
                    du[i] = -du[i]
                    du[i+1] = -du[i+1]
                    du[i+2] = -du[i+2]
            for i in range(12):
                if j+k == 1:
                    u[i] = u[i] - du[i]
                else:
                    u[i] = u[i] + du[i]

    # image-source contribution
    d = depth - z
    p = y*cd + d*sd
    q = y*sd - d*cd
    et[0] = p-aw1
    et[1] = p-aw2
    if abs(q) < eps:
        q = 0
    if abs(et[0]) < eps:
        et[0] = 0
    if abs(et[1]) < eps:
        et[1] = 0

    # reject singular case
    ## on fault edge
    if (q == 0 and ((xi[0]*xi[1] <= 0 and et[0]*et[1]==0) or (et[0]*et[1]<=0 and xi[0]*xi[1] == 0))):
        print('Singular case on fault edge')
        return 0,0,0,0,0,0,0,0,0,0,0,0
    ## on negative extension of fault edge
    kxi[0] = 0
    kxi[1] = 0
    ket[0] = 0
    ket[1] = 0
    r12 = np.sqrt(xi[0]*xi[0] + et[1]*et[1] + q*q)
    r21 = np.sqrt(xi[1]*xi[1] + et[0]*et[0] + q*q)
    r22 = np.sqrt(xi[1]*xi[1] + et[1]*et[1] + q*q)
    if (xi[0]<0 and r21+xi[1] < eps):
        kxi[0] = 1
    if (xi[0]<0 and r22+xi[1] < eps):
        kxi[1] = 1  
    if (et[0]<0 and r12+et[1] < eps):
        ket[0] = 1
    if (et[0]<0 and r22+et[1] < eps):
        ket[1] = 1
    for k in range(2):
        for j in range(2):
            c2 = DCCON2(xi[j],et[k],q,sd,cd,kxi[k],ket[j])
            dua = UA(xi[j],et[k],q,disl1,disl2,disl3,c0,c2)
            dub = UB(xi[j],et[k],q,disl1,disl2,disl3,c0,c2)
            duc = UC(xi[j],et[k],q,z,disl1,disl2,disl3,c0,c2)

            for i in range(0,10,3):
                du[i] = dua[i] + dub[i] + z*duc[i]
                du[i+1] = (dua[i+1]+dub[i+1]+z*duc[i+1])*cd  - (dua[i+2]+dub[i+2]+z*duc[i+2])*sd
                du[i+2] = (dua[i+1]+dub[i+1]-z*duc[i+1])*sd  + (dua[i+2]+dub[i+2]-z*duc[i+2])*cd
                if i == 9:
                    du[9] = du[9] + duc[0]
                    du[10] = du[10] + duc[1]*cd - duc[2]*sd
                    du[11] = du[11] - duc[1]*sd - duc[2]*cd
            for i in range(12):
                if j+k == 1:
                    u[i] = u[i] - du[i]
                else:
                    u[i] = u[i] + du[i]
    ux = u[0]
    uy = u[1]
    uz = u[2]
    uxx = u[3]
    uyx = u[4]
    uzx = u[5]
    uxy = u[6]
    uyy = u[7]
    uzy = u[8]
    uxz = u[9]
    uyz = u[10]
    uzz = u[11]

    return ux,uy,uz,uxx,uyx,uzx,uxy,uyy,uzy,uxz,uyz,uzz