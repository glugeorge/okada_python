import numpy as np

def UA0(x,y,d,pot1,pot2,pot3,pot4,c0,c1):
    ## Unpack variables
    # from DCCON0
    alp1 = c0['alp1']
    alp2 = c0['alp2']
    alp3 = c0['alp3']
    alp4 = c0['alp4']
    alp5 = c0['alp5']
    sd   = c0['sd']
    cd   = c0['cd']
    sdsd = c0['sdsd']
    cdcd = c0['cdcd']
    sdcd = c0['sdcd']
    s2d  = c0['s2d']
    c2d  = c0['c2d']

    # from DCCON1
    p   = c1['p']
    q   = c1['q']
    s   = c1['s']
    t   = c1['t']
    xy  = c1['xy']
    x2  = c1['x2']
    y2  = c1['y2']
    d2  = c1['d2']
    r   = c1['r']
    r2  = c1['r2']
    r3  = c1['r3']
    r5  = c1['r5']
    qr  = c1['qr']
    qrx = c1['qrx']
    a3  = c1['a3']
    a5  = c1['a5']
    b3  = c1['b3']
    c3  = c1['c3']
    uy  = c1['uy']
    vy  = c1['vy']
    wy  = c1['wy']
    uz  = c1['uz']
    vz  = c1['vz']
    wz  = c1['wz']
    n_x, n_y, n_z = x.shape
    u = np.zeros((n_x, n_y, n_z,12))
    du = np.zeros((n_x, n_y, n_z,12))

    # strike slip contribution
    if pot1 != 0:
        du[..., 0]  = alp1*q/r3    + alp2*x2*qr
        du[..., 1]  = alp1*x/r3*sd + alp2*xy*qr
        du[..., 2]  = -alp1*x/r3*cd + alp2*x*d*qr
        du[..., 3]  = x*qr*(-alp1 + alp2*(1 + a5))
        du[..., 4]  = alp1*a3/r3*sd + alp2*y*qr*a5
        du[..., 5]  = -alp1*a3/r3*cd + alp2*d*qr*a5
        du[..., 6]  = alp1*(sd/r3 - y*qr) + alp2*3*x2/r5*uy
        du[..., 7]  = 3*x/r5*(-alp1*y*sd + alp2*(y*uy + q))
        du[..., 8]  = 3*x/r5*(alp1*y*cd + alp2*d*uy)
        du[..., 9]  = alp1*(cd/r3 + d*qr) + alp2*3*x2/r5*uz
        du[..., 10] = 3*x/r5*(alp1*d*sd + alp2*y*uz)
        du[..., 11] = 3*x/r5*(-alp1*d*cd + alp2*(d*uz - q))
        u = u + pot1/(2*np.pi)*du
    # dip slip contribution
    if pot2 != 0:
        du[..., 0]  = alp2*x*p*qr
        du[..., 1]  = alp1*s/r3 + alp2*y*p*qr
        du[..., 2]  = -alp1*t/r3 + alp2*d*p*qr
        du[..., 3]  = alp2*p*qr*a5
        du[..., 4]  = -alp1*3*x*s/r5 - alp2*y*p*qrx
        du[..., 5]  = alp1*3*x*t/r5 - alp2*d*p*qrx
        du[..., 6]  = alp2*3*x/r5*vy
        du[..., 7]  = alp1*(s2d/r3 - 3*y*s/r5) + alp2*(3*y/r5*vy + p*qr)
        du[..., 8]  = -alp1*(c2d/r3 - 3*y*t/r5) + alp2*3*d/r5*vy
        du[..., 9]  = alp2*3*x/r5*vz
        du[..., 10] = alp1*(c2d/r3 + 3*d*s/r5) + alp2*3*y/r5*vz
        du[..., 11] = alp1*(s2d/r3 - 3*d*t/r5) + alp2*(3*d/r5*vz - p*qr)
        u = u + pot2/(2*np.pi)*du
    # tensile-fault contribution
    if pot3 != 0:
        du[..., 0]  = alp1*x/r3 - alp2*x*q*qr
        du[..., 1]  = alp1*t/r3 - alp2*y*q*qr
        du[..., 2]  = alp1*s/r3 - alp2*d*q*qr
        du[..., 3]  = alp1*a3/r3 - alp2*q*qr*a5
        du[..., 4]  = -alp1*3*x*t/r5 + alp2*y*q*qrx
        du[..., 5]  = -alp1*3*x*s/r5 + alp2*d*q*qrx
        du[..., 6]  = -alp1*3*xy/r5 - alp2*x*qr*wy
        du[..., 7]  = alp1*(c2d/r3 - 3*y*t/r5) - alp2*(y*wy + q)*qr
        du[..., 8]  = alp1*(s2d/r3 - 3*y*s/r5) - alp2*d*qr*wy
        du[..., 9]  = alp1*3*x*d/r5 - alp2*x*qr*wz
        du[..., 10] = -alp1*(s2d/r3 - 3*d*t/r5) - alp2*y*qr*wz
        du[..., 11] = alp1*(c2d/r3 + 3*d*s/r5) - alp2*(d*wz - q)*qr
        u = u + pot3/(2*np.pi)*du
    # inflate source contribution
    if pot4 != 0:
        du[..., 0]  = -alp1*x/r3
        du[..., 1]  = -alp1*y/r3
        du[..., 2]  = -alp1*d/r3
        du[..., 3]  = -alp1*a3/r3
        du[..., 4]  = alp1*3*xy/r5
        du[..., 5]  = alp1*3*x*d/r5
        du[..., 6]  = du[..., 4]
        du[..., 7]  = -alp1*b3/r3
        du[..., 8]  = alp1*3*y*d/r5
        du[..., 9]  = -du[..., 5]
        du[..., 10] = -du[..., 8]
        du[..., 11] = alp1*c3/r3
        u = u + pot4/(2*np.pi)*du  
    return u

def UB0(x,y,d,z,pot1,pot2,pot3,pot4,c0,c1):
    ## Unpack variables
    # from DCCON0
    alp1 = c0['alp1']
    alp2 = c0['alp2']
    alp3 = c0['alp3']
    alp4 = c0['alp4']
    alp5 = c0['alp5']
    sd   = c0['sd']
    cd   = c0['cd']
    sdsd = c0['sdsd']
    cdcd = c0['cdcd']
    sdcd = c0['sdcd']
    s2d  = c0['s2d']
    c2d  = c0['c2d']

    # from DCCON1
    p   = c1['p']
    q   = c1['q']
    s   = c1['s']
    t   = c1['t']
    xy  = c1['xy']
    x2  = c1['x2']
    y2  = c1['y2']
    d2  = c1['d2']
    r   = c1['r']
    r2  = c1['r2']
    r3  = c1['r3']
    r5  = c1['r5']
    qr  = c1['qr']
    qrx = c1['qrx']
    a3  = c1['a3']
    a5  = c1['a5']
    b3  = c1['b3']
    c3  = c1['c3']
    uy  = c1['uy']
    vy  = c1['vy']
    wy  = c1['wy']
    uz  = c1['uz']
    vz  = c1['vz']
    wz  = c1['wz']
    n_x, n_y, n_z = x.shape
    u = np.zeros((n_x, n_y, n_z,12))
    du = np.zeros((n_x, n_y, n_z,12))

    c = d + z
    rd = r + d
    d12 = 1/(r*rd*rd)
    d32 = d12*(2*r + d)/r2
    d33 = d12*(3*r + d)/(r2*rd)
    d53 = d12*(8*r2 + 9*r*d + d2)/(r2*r2*rd)
    d54 = d12*(5*r2 + 4*r*d + d2)/r3*d12

    fi1 = y * (d12 - x2 * d33)
    fi2 = x * (d12 - y2 * d33)
    fi3 = x/r3 - fi2
    fi4 = -xy * d32
    fi5 = 1/(r*rd) - x2 * d32
    fj1 = -3 * xy * (d33 - x2 * d54)
    fj2 = 1/r3 - 3 * d12 + 3 * x2 * y2 * d54
    fj3 = a3/r3 - fj2
    fj4 = -3 * xy/r5 - fj1
    fk1 = -y * (d32 - x2 * d53)
    fk2 = -x * (d32 - y2 * d53)
    fk3 = -3 * x * d/r5 - fk2

    if pot1 != 0:
        du[..., 0]  = -x2*qr - alp3*fi1*sd
        du[..., 1]  = -xy*qr - alp3*fi2*sd
        du[..., 2]  = -c*x*qr - alp3*fi4*sd
        du[..., 3]  = -x*qr*(1 + a5) - alp3*1*sd
        du[..., 4]  = -y*qr*a5 - alp3*2*sd
        du[..., 5]  = -c*qr*a5 - alp3*3*sd
        du[..., 6]  = -3*x2/r5*uy - alp3*2*sd
        du[..., 7]  = -3*xy/r5*uy - x*qr - alp3*4*sd
        du[..., 8]  = -3*c*x/r5*uy - alp3*3*sd
        du[..., 9]  = -3*x2/r5*uz + alp3*3*sd
        du[..., 10] = -3*xy/r5*uz + alp3*3*sd
        du[..., 11] = 3*x/r5*(-c*uz + alp3*y*sd)
        u = u + pot1/(2*np.pi)*du
    if pot2 != 0:
        du[..., 0]  = -x*p*qr + alp3*fi3*sdcd
        du[..., 1]  = -y*p*qr + alp3*fi1*sdcd
        du[..., 2]  = -c*p*qr + alp3*5*sdcd
        du[..., 3]  = -p*qr*a5 + alp3*3*sdcd
        du[..., 4]  = y*p*qrx + alp3*1*sdcd
        du[..., 5]  = c*p*qrx + alp3*3*sdcd
        du[..., 6]  = -3*x/r5*vy + alp3*1*sdcd
        du[..., 7]  = -3*y/r5*vy - p*qr + alp3*2*sdcd
        du[..., 8]  = -3*c/r5*vy + alp3*1*sdcd
        du[..., 9]  = -3*x/r5*vz - alp3*3*sdcd
        du[..., 10] = -3*y/r5*vz - alp3*1*sdcd
        du[..., 11] = -3*c/r5*vz + alp3*a3/r3*sdcd
        u = u + pot2/(2*np.pi)*du
    if pot3 != 0:
        du[..., 0]  = x*q*qr - alp3*fi3*sd
        du[..., 1]  = y*q*qr - alp3*fi1*sd
        du[..., 2]  = c*q*qr - alp3*5*sd
        du[..., 3]  = q*qr*a5 - alp3*3*sd
        du[..., 4]  = -y*q*qrx - alp3*1*sd
        du[..., 5]  = -c*q*qrx - alp3*3*sd
        du[..., 6]  = x*qr*wy - alp3*1*sd
        du[..., 7]  = qr*(y*wy + q) - alp3*2*sd
        du[..., 8]  = c*qr*wy - alp3*1*sd
        du[..., 9]  = x*qr*wz + alp3*3*sd
        du[..., 10] = y*qr*wz + alp3*1*sd
        du[..., 11] = c*qr*wz - alp3*a3/r3*sd
        u = u + pot3/(2*np.pi)*du
    if pot4 != 0:
        du[..., 0]  = alp3*x/r3
        du[..., 1]  = alp3*y/r3
        du[..., 2]  = alp3*d/r3
        du[..., 3]  = alp3*a3/r3
        du[..., 4]  = -alp3*3*xy/r5
        du[..., 5]  = -alp3*3*x*d/r5
        du[..., 6]  = du[..., 4]
        du[..., 7]  = alp3*b3/r3
        du[..., 8]  = -alp3*3*y*d/r5
        du[..., 9]  = -du[..., 5]
        du[..., 10] = -du[..., 8]
        du[..., 11] = -alp3*c3/r3
        u = u + pot4/(2*np.pi)*du
    return u

def UC0(x,y,d,z,pot1,pot2,pot3,pot4,c0,c1):
    ## Unpack variables
    # from DCCON0
    alp1 = c0['alp1']
    alp2 = c0['alp2']
    alp3 = c0['alp3']
    alp4 = c0['alp4']
    alp5 = c0['alp5']
    sd   = c0['sd']
    cd   = c0['cd']
    sdsd = c0['sdsd']
    cdcd = c0['cdcd']
    sdcd = c0['sdcd']
    s2d  = c0['s2d']
    c2d  = c0['c2d']

    # from DCCON1
    p   = c1['p']
    q   = c1['q']
    s   = c1['s']
    t   = c1['t']
    xy  = c1['xy']
    x2  = c1['x2']
    y2  = c1['y2']
    d2  = c1['d2']
    r   = c1['r']
    r2  = c1['r2']
    r3  = c1['r3']
    r5  = c1['r5']
    qr  = c1['qr']
    qrx = c1['qrx']
    a3  = c1['a3']
    a5  = c1['a5']
    b3  = c1['b3']
    c3  = c1['c3']
    uy  = c1['uy']
    vy  = c1['vy']
    wy  = c1['wy']
    uz  = c1['uz']
    vz  = c1['vz']
    wz  = c1['wz']
    n_x, n_y, n_z = x.shape
    u = np.zeros((n_x, n_y, n_z,12))
    du = np.zeros((n_x, n_y, n_z,12))

    c = d + z
    q2 = q * q
    r7 = r5 * r2
    a7 = 1 - 7 * x2 / r2
    b5 = 1 - 5 * y2 / r2
    b7 = 1 - 7 * y2 / r2
    c5 = 1 - 5 * d2 / r2
    c7 = 1 - 7 * d2 / r2
    d7 = 2 - 7 * q2 / r2
    qr5 = 5 * q / r2
    qr7 = 7 * q / r2
    dr5 = 5 * d / r2

    if pot1 != 0:
        du[..., 0]= -alp4*a3/r3*cd + alp5*c*qr*a5
        du[..., 1]= 3*x/r5*(alp4*y*cd + alp5*c*(sd-y*qr5))
        du[..., 2]= 3*x/r5*(-alp4*y*sd + alp5*c*(cd+d*qr5))
        du[..., 3]= alp4*3*x/r5*(2+a5)*cd - alp5*c*qrx*(2+a7)
        du[..., 4]= 3/r5*(alp4*y*a5*cd + alp5*c*(a5*sd-y*qr5*a7))
        du[..., 5]= 3/r5*(-alp4*y*a5*sd + alp5*c*(a5*cd+d*qr5*a7))
        du[..., 6]= du[..., 4]
        du[..., 7]= 3*x/r5*(alp4*b5*cd - alp5*5*c/r2*(2*y*sd+q*b7))
        du[..., 8]= 3*x/r5*(-alp4*b5*sd + alp5*5*c/r2*(d*b7*sd-y*c7*cd))
        du[..., 9]= 3/r5*(-alp4*d*a5*cd + alp5*c*(a5*cd+d*qr5*a7))
        du[..., 10]= 15*x/r7*(alp4*y*d*cd + alp5*c*(d*b7*sd-y*c7*cd))
        du[..., 11]= 15*x/r7*(-alp4*y*d*sd + alp5*c*(2*d*cd-q*c7))
        u = u + pot1/(2*np.pi)*du
    if pot2 != 0:
        du[..., 0]= alp4*3*x*t/r5 - alp5*c*p*qrx
        du[..., 1]= -alp4/r3*(c2d-3*y*t/r2) + alp5*3*c/r5*(s-y*p*qr5)
        du[..., 2]= -alp4*a3/r3*sdcd + alp5*3*c/r5*(t+d*p*qr5)
        du[..., 3]= alp4*3*t/r5*a5 - alp5*5*c*p*qr/r2*a7
        du[..., 4]= 3*x/r5*(alp4*(c2d-5*y*t/r2)-alp5*5*c/r2*(s-y*p*qr7))
        du[..., 5]= 3*x/r5*(alp4*(2+a5)*sdcd - alp5*5*c/r2*(t+d*p*qr7))
        du[..., 6]= du[..., 4]
        du[..., 7]= 3/r5*(alp4*(2*y*c2d+t*b5) + alp5*c*(s2d-10*y*s/r2-p*qr5*b7))
        du[..., 8]= 3/r5*(alp4*y*a5*sdcd - alp5*c*((3+a5)*c2d+y*p*dr5*qr7))
        du[..., 9]= 3*x/r5*(-alp4*(s2d-t*dr5) - alp5*5*c/r2*(t+d*p*qr7))
        du[..., 10]= 3/r5*(-alp4*(d*b5*c2d+y*c5*s2d) - alp5*c*((3+a5)*c2d+y*p*dr5*qr7))
        du[..., 11]= 3/r5*(-alp4*d*a5*sdcd - alp5*c*(s2d-10*d*t/r2+p*qr5*c7))
        u = u + pot2/(2*np.pi)*du
    if pot3 != 0:
        du[..., 0]= 3*x/r5*(-alp4*s + alp5*(c*q*qr5-z))
        du[..., 1]= alp4/r3*(s2d-3*y*s/r2) + alp5*3/r5*(c*(t-y+y*q*qr5)-y*z)
        du[..., 2]= -alp4/r3*(1-a3*sdsd) - alp5*3/r5*(c*(s-d+d*q*qr5)-d*z)
        du[..., 3]= -alp4*3*s/r5*a5 + alp5*(c*qr*qr5*a7-3*z/r5*a5)
        du[..., 4]= 3*x/r5*(-alp4*(s2d-5*y*s/r2) - alp5*5/r2*(c*(t-y+y*q*qr7)-y*z))
        du[..., 5]= 3*x/r5*(alp4*(1-(2+a5)*sdsd) + alp5*5/r2*(c*(s-d+d*q*qr7)-d*z))
        du[..., 6]= du[..., 4]
        du[..., 7]= 3/r5*(-alp4*(2*y*s2d+s*b5) - alp5*(c*(2*sdsd+10*y*(t-y)/r2-q*qr5*b7)+z*b5))
        du[..., 8]= 3/r5*(alp4*y*(1-a5*sdsd) + alp5*(c*(3+a5)*s2d-y*dr5*(c*d7+z)))
        du[..., 9]= 3*x/r5*(-alp4*(c2d+s*dr5) + alp5*(5*c/r2*(s-d+d*q*qr7)-1-z*dr5))
        du[..., 10]= 3/r5*(alp4*(d*b5*s2d-y*c5*c2d) + alp5*(c*((3+a5)*s2d-y*dr5*d7)-y*(1+z*dr5)))
        du[..., 11]= 3/r5*(-alp4*d*(1-a5*sdsd) - alp5*(c*(c2d+10*d*(s-d)/r2-q*qr5*c7)+z*(1+c5)))
        u = u + pot3/(2*np.pi)*du
    if pot4 != 0:
        du[..., 0]= alp4*3*x*d/r5
        du[..., 1]= alp4*3*y*d/r5
        du[..., 2]= alp4*c3/r3
        du[..., 3]= alp4*3*d/r5*a5
        du[..., 4]= -alp4*15*xy*d/r7
        du[..., 5]= -alp4*3*x/r5*c5
        du[..., 6]= du[..., 4]
        du[..., 7]= alp4*3*d/r5*b5
        du[..., 8]= -alp4*3*y/r5*c5
        du[..., 9]= du[..., 5]
        du[..., 10]= alp4*3*d/r5*(2+c5)
        u = u + pot4/(2*np.pi)*du
    return u

def UA(xi,et,q,disl1,disl2,disl3,c0,c2):
    ## UNPACK VARIABLES
    # From DCCON0
    sd = c0['sd']
    cd = c0['cd']
    alp1 = c0['alp1']
    alp2 = c0['alp2']
    
    # From DCCON2
    xi2 = c2['xi2']
    q2 = c2['q2']
    r = c2['r']
    r3 = c2['r3']
    y = c2['y']
    d = c2['d']
    tt = c2['tt']
    alx = c2['alx']
    ale = c2['ale']
    x11 = c2['x11']
    y11 = c2['y11']
    y32 = c2['y32']
    ey = c2['ey']
    ez = c2['ez']
    fy = c2['fy']
    fz = c2['fz']
    gy = c2['gy']
    gz = c2['gz']
    hy = c2['hy']
    hz = c2['hz']
    
    xy = xi*y11
    qx = q*x11
    qy = q*y11
    n_x, n_y, n_z = xi.shape
    u = np.zeros((n_x, n_y, n_z,12))
    du = np.zeros((n_x, n_y, n_z,12))
    # strike-slip contribution
    if disl1 != 0:
        du[...,0] = tt/2 + alp2*xi*qy
        du[...,1] = alp2*q/r
        du[...,2] = alp1*ale -alp2*q*qy
        du[...,3] = -alp1*qy  -alp2*xi2*q*y32
        du[...,4] = -alp2*xi*q/r3
        du[...,5] = alp1*xy  +alp2*xi*q2*y32
        du[...,6] = alp1*xy*sd + alp2*xi*fy+d/2*x11
        du[...,7] = alp2*ey
        du[...,8] = alp1*(cd/r+qy*sd) -alp2*q*fy
        du[...,9] = alp1*xy*cd + alp2*xi*fz+y/2*x11
        du[...,10] = alp2*ez
        du[...,11] = -alp1*(sd/r-qy*cd) -alp2*q*fz 
        u = u + disl1/(2*np.pi)*du

    # dip-slip contribution
    if disl2 != 0:
        du[...,0] = alp2*q/r
        du[...,1] = tt/2 + alp2*et*qx
        du[...,2] = alp1*alx -alp2*q*qx
        du[...,3] = -alp2*xi*q/r3
        du[...,4] = -qy/2 -alp2*et*q/r3
        du[...,5] = alp1/r + alp2*q2/r3
        du[...,6] = alp2*ey
        du[...,7] = alp1*d*x11+xy/2*sd +alp2*et*gy
        du[...,8] = alp1*y*x11 - alp2*q*gy
        du[...,9] = alp2*ez
        du[...,10] = alp1*y*x11+xy/2*cd + alp2*et*gz
        du[...,11] = -alp1*d*x11 - alp2*q*gz
        u = u + disl2/(2*np.pi)*du

    # tensile-fault contribution
    if disl3 != 0:
        du[...,0] = -alp1*ale -alp2*q*qy
        du[...,1] =-alp1*alx -alp2*q*qx
        du[...,2] =    tt/2 -alp2*(et*qx+xi*qy)
        du[...,3] =-alp1*xy  +alp2*xi*q2*y32
        du[...,4] =-alp1/r   +alp2*q2/r3 
        du[...,5] =-alp1*qy  -alp2*q*q2*y32
        du[...,6] =-alp1*(cd/r+qy*sd)  -alp2*q*fy
        du[...,7] =-alp1*y*x11         -alp2*q*gy
        du[...,8] = alp1*(d*x11+xy*sd) +alp2*q*hy
        du[...,9] = alp1*(sd/r-qy*cd)  -alp2*q*fz
        du[...,10] = alp1*d*x11         -alp2*q*gz
        du[...,11] = alp1*(y*x11+xy*cd) +alp2*q*hz
        u = u + disl3/(2*np.pi)*du
        
    return u

def UB(xi,et,q,disl1,disl2,disl3,c0,c2):
    # unpack c0
    cd = c0['cd']
    sd = c0['sd']
    cdcd = c0['cdcd']
    sdcd = c0['sdcd']
    sdsd = c0['sdsd']
    alp3 = c0['alp3']
    # unpack c2
    xi2 = c2['xi2']
    q2 = c2['q2']
    r = c2['r']
    r3 = c2['r3']
    d = c2['d']
    y = c2['y']
    ale = c2['ale']
    tt = c2['tt']
    x11 = c2['x11']
    y11 = c2['y11']
    y32 = c2['y32']
    ey = c2['ey']
    ez = c2['ez']
    fy = c2['fy']
    fz = c2['fz']
    gy = c2['gy']
    gz = c2['gz']
    hy = c2['hy']
    hz = c2['hz']

    rd = r+d
    d11 = 1/(r*rd)
    aj2 = xi*y/rd*d11
    aj5 = -(d+y*y/rd)*d11
    if cd != 0:
        x = np.sqrt(xi2+q2)
        ai4 = np.where(xi==0,0,1/cdcd * (xi/rd*sdcd + 2*np.arctan((et*(x+q*cd)+x*(r+x)*sd)/(xi*(r+x)*cd))))
        ai3 = (y*cd/rd-ale+sd*np.log(rd))/cdcd
        ak1 = xi*(d11-y11*sd)/cd
        ak3 = (q*y11-y*d11)/cd
        aj3 = (ak1-aj2*sd)/cd
        aj6 = (ak3-aj5*sd)/cd
    else:
        rd2 = rd*rd
        ai3 = (et/rd + y*q/rd2 - ale)/2
        ai4 = xi*y/rd2/2
        ak1 = xi*q/rd*d11
        ak3 = sd/rd*(xi2*d11-1)
        aj3 = -xi/rd2*(q2*d11-1/2)
        aj6 = -y/rd2*(xi2*d11-1/2)
    xy = xi*y11
    ai1 = -xi/rd*cd - ai4*sd
    ai2 = np.log(rd)+ai3*sd
    ak2 = 1/r + ak3*sd
    ak4 = xy*cd - ak1*sd
    aj1 = aj5*cd - aj6*sd
    aj4 = -xy-aj2*cd+aj3*sd
    n_x, n_y, n_z = xi.shape
    u = np.zeros((n_x, n_y, n_z,12))
    du = np.zeros((n_x, n_y, n_z,12))
    qx = q*x11
    qy = q*y11
    # strike-slip contribution
    if disl1 != 0:
        du[...,0] = -xi*qy-tt - alp3*ai1*sd
        du[...,1] = -q/r      +alp3*y/rd*sd
        du[...,2] = q*qy     -alp3*ai2*sd
        du[...,3] = xi2*q*y32 -alp3*aj1*sd
        du[...,4] = xi*q/r3   -alp3*aj2*sd
        du[...,5] = -xi*q2*y32 -alp3*aj3*sd
        du[...,6] = -xi*fy-d*x11 +alp3*(xy+aj4)*sd
        du[...,7] = -ey          +alp3*(1/r+aj5)*sd
        du[...,8] = q*fy        -alp3*(qy-aj6)*sd
        du[...,9] = -xi*fz-y*x11 +alp3*ak1*sd
        du[...,10] = -ez          +alp3*y*d11*sd
        du[...,11] = q*fz        +alp3*ak2*sd
        u = u + disl1/(2*np.pi)*du
            
    # dip-slip contribution
    if disl2 != 0:
        du[...,0]=-q/r      +alp3*ai3*sdcd
        du[...,1]=-et*qx-tt -alp3*xi/rd*sdcd
        du[...,2]= q*qx     +alp3*ai4*sdcd
        du[...,3]= xi*q/r3     +alp3*aj4*sdcd
        du[...,4]= et*q/r3+qy  +alp3*aj5*sdcd
        du[...,5]=-q2/r3       +alp3*aj6*sdcd
        du[...,6]=-ey          +alp3*aj1*sdcd
        du[...,7]=-et*gy-xy*sd +alp3*aj2*sdcd
        du[...,8]= q*gy        +alp3*aj3*sdcd
        du[...,9]=-ez          -alp3*ak3*sdcd
        du[...,10]=-et*gz-xy*cd -alp3*xi*d11*sdcd
        du[...,11]= q*gz        -alp3*ak4*sdcd
        u = u + disl2/(2*np.pi)*du

    # tensile fault contribution
    if disl3 != 0:
        du[...,0]= q*qy           -alp3*ai3*sdsd
        du[...,1]= q*qx           +alp3*xi/rd*sdsd
        du[...,2]= et*qx+xi*qy-tt -alp3*ai4*sdsd
        du[...,3]=-xi*q2*y32 -alp3*aj4*sdsd
        du[...,4]=-q2/r3     -alp3*aj5*sdsd
        du[...,5]= q*q2*y32  -alp3*aj6*sdsd
        du[...,6]= q*fy -alp3*aj1*sdsd
        du[...,7]= q*gy -alp3*aj2*sdsd
        du[...,8]=-q*hy -alp3*aj3*sdsd
        du[...,9]= q*fz +alp3*ak3*sdsd
        du[...,10]= q*gz +alp3*xi*d11*sdsd
        du[...,11]=-q*hz +alp3*ak4*sdsd
        u = u + disl3/(2*np.pi)*du
    return u
    
def UC(xi,et,q,z,disl1,disl2,disl3,c0,c2):
    # unpack c0
    alp4 = c0['alp4']
    alp5 = c0['alp5']
    sd = c0['sd']
    cd = c0['cd']
    sdsd = c0['sdsd']
    sdcd = c0['sdcd']
    cdcd = c0['cdcd']
    
    # unpack c2
    xi2 = c2['xi2']
    et2 = c2['et2']
    q2 = c2['q2']
    y = c2['y']
    d = c2['d']
    r = c2['r']
    r2 = c2['r2']
    r3 = c2['r3']
    r5 = c2['r5']
    x11 = c2['x11']
    y11 = c2['y11']
    x32 = c2['x32']
    y32 = c2['y32']

    c=d+z
    x53=(8*r2+9*r*xi+3*xi2)*x11*x11*x11/r2
    y53=(8*r2+9*r*et+3*et2)*y11*y11*y11/r2
    h=q*cd-z
    z32=sd/r3-h*y32
    z53=3*sd/r5-h*y53
    y0=y11-xi2*y32
    z0=z32-xi2*z53
    ppy=cd/r3+q*y32*sd
    ppz=sd/r3-q*y32*cd
    qq=z*y32+z32+z0
    qqy=3*c*d/r5-qq*sd
    qqz=3*c*y/r5-qq*cd+q*y32
    xy=xi*y11
    qy=q*y11
    qr=3*q/r5
    cdr=(c+d)/r3
    yy0=y/r3-y0*cd
    n_x, n_y, n_z = xi.shape
    u = np.zeros((n_x, n_y, n_z,12))
    du = np.zeros((n_x, n_y, n_z,12))

    # strike-slip contribution
    if disl1 != 0:
        du[...,0]= alp4*xy*cd           -alp5*xi*q*z32
        du[...,1]= alp4*(cd/r+2*qy*sd) -alp5*c*q/r3
        du[...,2]= alp4*qy*cd           -alp5*(c*et/r3-z*y11+xi2*z32)
        du[...,3]= alp4*y0*cd                  -alp5*q*z0
        du[...,4]=-alp4*xi*(cd/r3+2*q*y32*sd) +alp5*c*xi*qr
        du[...,5]=-alp4*xi*q*y32*cd            +alp5*xi*(3*c*et/r5-qq)
        du[...,6]=-alp4*xi*ppy*cd    -alp5*xi*qqy
        du[...,7]= alp4*2*(d/r3-y0*sd)*sd-y/r3*cd - alp5*(cdr*sd-et/r3-c*y*qr)
        du[...,8]=-alp4*q/r3+yy0*sd  +alp5*(cdr*cd+c*d*qr-(y0*cd+q*z0)*sd)
        du[...,9]= alp4*xi*ppz*cd    -alp5*xi*qqz
        du[...,10]= alp4*2*(y/r3-y0*cd)*sd+d/r3*cd -alp5*(cdr*cd+c*d*qr)
        du[...,11]=         yy0*cd    -alp5*(cdr*sd-c*y*qr-y0*sdsd+q*z0*cd)
        u = u + disl1/(2*np.pi)*du
    
    # dip-slip contribution
    if disl2 != 0:
        du[...,0]= alp4*cd/r -qy*sd -alp5*c*q/r3
        du[...,1]= alp4*y*x11       -alp5*c*et*q*x32
        du[...,2]=     -d*x11-xy*sd -alp5*c*(x11-q2*x32)
        du[...,3]=-alp4*xi/r3*cd +alp5*c*xi*qr +xi*q*y32*sd
        du[...,4]=-alp4*y/r3     +alp5*c*et*qr
        du[...,5]=    d/r3-y0*sd +alp5*c/r3*(1-3*q2/r2)
        du[...,6]=-alp4*et/r3+y0*sdsd -alp5*(cdr*sd-c*y*qr)
        du[...,7]= alp4*(x11-y*y*x32) -alp5*c*((d+2*q*cd)*x32-y*et*q*x53)
        du[...,8]=  xi*ppy*sd+y*d*x32 +alp5*c*((y+2*q*sd)*x32-y*q2*x53)
        du[...,9]=      -q/r3+y0*sdcd -alp5*(cdr*cd+c*d*qr)
        du[...,10]= alp4*y*d*x32       -alp5*c*((y-2*q*sd)*x32+d*et*q*x53)
        du[...,11]=-xi*ppz*sd+x11-d*d*x32-alp5*c*((d-2*q*cd)*x32-d*q2*x53)
        u = u + disl2/(2*np.pi)*du
    
    if disl3 != 0:
        du[...,0]=-alp4*(sd/r+qy*cd)   -alp5*(z*y11-q2*z32)
        du[...,1]= alp4*2*xy*sd+d*x11 -alp5*c*(x11-q2*x32)
        du[...,2]= alp4*(y*x11+xy*cd)  +alp5*q*(c*et*x32+xi*z32)
        du[...,3]= alp4*xi/r3*sd+xi*q*y32*cd+alp5*xi*(3*c*et/r5-2*z32-z0)
        du[...,4]= alp4*2*y0*sd-d/r3 +alp5*c/r3*(1-3*q2/r2)
        du[...,5]=-alp4*yy0           -alp5*(c*et*qr-q*z0)
        du[...,6]= alp4*(q/r3+y0*sdcd)   +alp5*(z/r3*cd+c*d*qr-q*z0*sd)
        du[...,7]=-alp4*2*xi*ppy*sd-y*d*x32 +alp5*c*((y+2*q*sd)*x32-y*q2*x53)
        du[...,8]=-alp4*(xi*ppy*cd-x11+y*y*x32) +alp5*(c*((d+2*q*cd)*x32-y*et*q*x53)+xi*qqy)
        du[...,9]=  -et/r3+y0*cdcd -alp5*(z/r3*sd-c*y*qr-y0*sdsd+q*z0*cd)
        du[...,10]= alp4*2*xi*ppz*sd-x11+d*d*x32-alp5*c*((d-2*q*cd)*x32-d*q2*x53)
        du[...,11]= alp4*(xi*ppz*cd+y*d*x32)+alp5*(c*((y-2*q*sd)*x32+d*et*q*x53)+xi*qqz)
        u = u + disl3/(2*np.pi)*du

    return u