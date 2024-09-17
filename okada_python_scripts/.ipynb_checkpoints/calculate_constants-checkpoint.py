import numpy as np
def DCCON0(alpha,dip):
    eps = 1e-6
    alp1 = (1-alpha)/2
    alp2 = alpha/2
    alp3 = (1-alpha)/alpha
    alp4 = 1-alpha
    alp5 = alpha
    dip_rad = np.deg2rad(dip)
    sd = np.sin(dip_rad)
    cd = np.cos(dip_rad)
    if abs(cd) < eps:
        cd = 0
        if sd > 0:
            sd = 1
        if sd < 0:
            sd = -1
    sdsd = sd*sd
    cdcd = cd*cd
    sdcd = sd*cd
    s2d = 2*sdcd
    c2d = cdcd-sdsd
    c0 = dict(alp1 = alp1, alp2 = alp2, alp3 = alp3, alp4 = alp4, alp5 = alp5, 
              sd = sd, cd = cd, sdsd = sdsd, cdcd = cdcd, sdcd = sdcd, s2d = s2d, c2d = c2d)
    return c0

def DCCON2(xi,et,q,sd,cd,kxi,ket):
    eps = 1e-6
    if abs(xi) < eps:
        xi = 0
    if abs(et) < eps:
        et = 0
    if abs(q) < eps:
        q = 0
    xi2 = xi*xi
    et2 = et*et
    q2 = q*q
    r2 = xi2+et2+q2
    r = np.sqrt(r2)
    if r == 0:
        c2 = dict(xi2 = xi2, et2 = et2, q2 = q2, r = r, r2 = r2, r3 = 0, r5 = 0, 
              y = 0, d = 0, tt = 0, alx = 0, ale = 0, 
              x11 = 0, y11 = 0, x32 = 0, y32 = 0, 
              ey = 0, ez = 0, fy = 0, fz = 0, gy = 0, gz = 0, hy = 0, hz = 0)
        return c2
    r3 = r*r2
    r5 = r3*r2
    y = et*cd + q*sd
    d = et*sd - q*cd

    if q == 0:
        tt = 0
    else:
        tt = np.arctan(xi*et/(q*r))

    if kxi == 1:
        alx = -np.log(r-xi)
        x11 = 0
        x32 = 0
    else:
        rxi = r+xi
        alx = np.log(rxi)
        x11 = 1/(r*rxi)
        x32 = (r+rxi)*x11*x11/r

    if ket == 1:
        ale = -np.log(r-et)
        y11 = 0
        y32 = 0
    else:
        ret = r+et
        ale = np.log(ret)
        y11 = 1/(r*ret)
        y32 = (r+ret)*y11*y11/r
    
    ey=sd/r-y*q/r3
    ez=cd/r+d*q/r3
    fy=d/r3+xi2*y32*sd
    fz=y/r3+xi2*y32*cd
    gy=2*x11*sd-y*q*x32
    gz=2*x11*cd+d*q*x32
    hy=d*q*x32+xi*q*y32*sd
    hz=y*q*x32+xi*q*y32*cd
    c2 = dict(xi2 = xi2, et2 = et2, q2 = q2, r = r, r2 = r2, r3 = r3, r5 = r5, 
              y = y, d = d, tt = tt, alx = alx, ale = ale, 
              x11 = x11, y11 = y11, x32 = x32, y32 = y32, 
              ey = ey, ez = ez, fy = fy, fz = fz, gy = gy, gz = gz, hy = hy, hz = hz)

    return c2 