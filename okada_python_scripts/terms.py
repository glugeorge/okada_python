import numpy as np

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
    u = np.zeros(12)
    du = np.zeros(12)
    # strike-slip contribution
    if disl1 != 0:
        du[0] = tt/2 + alp2*xi*qy
        du[1] = alp2*q/r
        du[2] = alp1*ale -alp2*q*qy
        du[3] = -alp1*qy  -alp2*xi2*q*y32
        du[4] = -alp2*xi*q/r3
        du[5] = alp1*xy  +alp2*xi*q2*y32
        du[6] = alp1*xy*sd + alp2*xi*fy+d/2*x11
        du[7] = alp2*ey
        du[8] = alp1*(cd/r+qy*sd) -alp2*q*fy
        du[9] = alp1*xy*cd + alp2*xi*fz+y/2*x11
        du[10] = alp2*ez
        du[11] = -alp1*(sd/r-qy*cd) -alp2*q*fz 
        for i in range(12):
            u[i] = u[i]+disl1/(2*np.pi)*du[i]
    # dip-slip contribution
    if disl2 != 0:
        du[0] = alp2*q/r
        du[1] = tt/2 + alp2*et*qx
        du[2] = alp1*alx -alp2*q*qx
        du[3] = -alp2*xi*q/r3
        du[4] = -qy/2 -alp2*et*q/r3
        du[5] = alp1/r + alp2*q2/r3
        du[6] = alp2*ey
        du[7] = alp1*d*x11+xy/2*sd +alp2*et*gy
        du[8] = alp1*y*x11 - alp2*q*gy
        du[9] = alp2*ez
        du[10] = alp1*y*x11+xy/2*cd + alp2*et*gz
        du[11] = -alp1*d*x11 - alp2*q*gz
        for i in range(12):
            u[i] = u[i]+disl2/(2*np.pi)*du[i]
    # tensile-fault contribution
    if disl3 != 0:
        du[0] = -alp1*ale -alp2*q*qy
        du[1] =-alp1*alx -alp2*q*qx
        du[2] =    tt/2 -alp2*(et*qx+xi*qy)
        du[3] =-alp1*xy  +alp2*xi*q2*y32
        du[4] =-alp1/r   +alp2*q2/r3 
        du[5] =-alp1*qy  -alp2*q*q2*y32
        du[6] =-alp1*(cd/r+qy*sd)  -alp2*q*fy
        du[7] =-alp1*y*x11         -alp2*q*gy
        du[8] = alp1*(d*x11+xy*sd) +alp2*q*hy
        du[9] = alp1*(sd/r-qy*cd)  -alp2*q*fz
        du[10] = alp1*d*x11         -alp2*q*gz
        du[11] = alp1*(y*x11+xy*cd) +alp2*q*hz
        for i in range(12):
            u[i] = u[i] + disl3/(2*np.pi)*du[i]
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
        if xi == 0:
            ai4 = 0
        else:
            x = np.sqrt(xi2+q2)
            ai4 = 1/cdcd * (xi/rd*sdcd + 2*np.arctan((et*(x+q*cd)+x*(r+x)*sd)/(xi*(r+x)*cd)))
        ai3 = (y+cd/rd-ale+sd*np.log(rd))/cdcd
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
    u = np.zeros(12)
    du = np.zeros(12)
    qx = q*x11
    qy = q*y11
    # strike-slip contribution
    if disl1 != 0:
        du[0] = -xi*qy-tt - alp3*ai1*sd
        du[1] = -q/r      +alp3*y/rd*sd
        du[2] = q*qy     -alp3*ai2*sd
        du[3] = xi2*q*y32 -alp3*aj1*sd
        du[4] = xi*q/r3   -alp3*aj2*sd
        du[5] = -xi*q2*y32 -alp3*aj3*sd
        du[6] = -xi*fy-d*x11 +alp3*(xy+aj4)*sd
        du[7] = -ey          +alp3*(1/r+aj5)*sd
        du[8] = q*fy        -alp3*(qy-aj6)*sd
        du[9] = -xi*fz-y*x11 +alp3*ak1*sd
        du[10] = -ez          +alp3*y*d11*sd
        du[11] = q*fz        +alp3*ak2*sd
        for i in range(12):
            u[i] = u[i] + disl1/(2*np.pi)*du[i]
            
    # dip-slip contribution
    if disl2 != 0:
        du[0]=-q/r      +alp3*ai3*sdcd
        du[1]=-et*qx-tt -alp3*xi/rd*sdcd
        du[2]= q*qx     +alp3*ai4*sdcd
        du[3]= xi*q/r3     +alp3*aj4*sdcd
        du[4]= et*q/r3+qy  +alp3*aj5*sdcd
        du[5]=-q2/r3       +alp3*aj6*sdcd
        du[6]=-ey          +alp3*aj1*sdcd
        du[7]=-et*gy-xy*sd +alp3*aj2*sdcd
        du[8]= q*gy        +alp3*aj3*sdcd
        du[9]=-ez          -alp3*ak3*sdcd
        du[10]=-et*gz-xy*cd -alp3*xi*d11*sdcd
        du[11]= q*gz        -alp3*ak4*sdcd
        for i in range(12):
            u[i] = u[i] + disl2/(2*np.pi)*du[i]

    # tensile fault contribution
    if disl3 != 0:
        du[0]= q*qy           -alp3*ai3*sdsd
        du[1]= q*qx           +alp3*xi/rd*sdsd
        du[2]= et*qx+xi*qy-tt -alp3*ai4*sdsd
        du[3]=-xi*q2*y32 -alp3*aj4*sdsd
        du[4]=-q2/r3     -alp3*aj5*sdsd
        du[5]= q*q2*y32  -alp3*aj6*sdsd
        du[6]= q*fy -alp3*aj1*sdsd
        du[7]= q*gy -alp3*aj2*sdsd
        du[8]=-q*hy -alp3*aj3*sdsd
        du[9]= q*fz +alp3*ak3*sdsd
        du[10]= q*gz +alp3*xi*d11*sdsd
        du[11]=-q*hz +alp3*ak4*sdsd
        for i in range(12):
            u[i] = u[i] + disl3/(2*np.pi)*du[i]
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
    qx=q*x11
    qy=q*y11
    qr=3*q/r5
    cqx=c*q*x53
    cdr=(c+d)/r3
    yy0=y/r3-y0*cd
    u = np.zeros(12)
    du = np.zeros(12)

    # strike-slip contribution
    if disl1 != 0:
        du[0]= alp4*xy*cd           -alp5*xi*q*z32
        du[1]= alp4*(cd/r+2*qy*sd) -alp5*c*q/r3
        du[2]= alp4*qy*cd           -alp5*(c*et/r3-z*y11+xi2*z32)
        du[3]= alp4*y0*cd                  -alp5*q*z0
        du[4]=-alp4*xi*(cd/r3+2*q*y32*sd) +alp5*c*xi*qr
        du[5]=-alp4*xi*q*y32*cd            +alp5*xi*(3*c*et/r5-qq)
        du[6]=-alp4*xi*ppy*cd    -alp5*xi*qqy
        du[7]= alp4*2*(d/r3-y0*sd)*sd-y/r3*cd - alp5*(cdr*sd-et/r3-c*y*qr)
        du[8]=-alp4*q/r3+yy0*sd  +alp5*(cdr*cd+c*d*qr-(y0*cd+q*z0)*sd)
        du[9]= alp4*xi*ppz*cd    -alp5*xi*qqz
        du[10]= alp4*2*(y/r3-y0*cd)*sd+d/r3*cd -alp5*(cdr*cd+c*d*qr)
        du[11]=         yy0*cd    -alp5*(cdr*sd-c*y*qr-y0*sdsd+q*z0*cd)
        for i in range(12):
            u[i] = u[i] + disl1/(2*np.pi)*du[i]
    
    # dip-slip contribution
    if disl2 != 0:
        du[0]= alp4*cd/r -qy*sd -alp5*c*q/r3
        du[1]= alp4*y*x11       -alp5*c*et*q*x32
        du[2]=     -d*x11-xy*sd -alp5*c*(x11-q2*x32)
        du[3]=-alp4*xi/r3*cd +alp5*c*xi*qr +xi*q*y32*sd
        du[4]=-alp4*y/r3     +alp5*c*et*qr
        du[5]=    d/r3-y0*sd +alp5*c/r3*(1-3*q2/r2)
        du[6]=-alp4*et/r3+y0*sdsd -alp5*(cdr*sd-c*y*qr)
        du[7]= alp4*(x11-y*y*x32) -alp5*c*((d+2*q*cd)*x32-y*et*q*x53)
        du[8]=  xi*ppy*sd+y*d*x32 +alp5*c*((y+2*q*sd)*x32-y*q2*x53)
        du[9]=      -q/r3+y0*sdcd -alp5*(cdr*cd+c*d*qr)
        du[10]= alp4*y*d*x32       -alp5*c*((y-2*q*sd)*x32+d*et*q*x53)
        du[11]=-xi*ppz*sd+x11-d*d*x32-alp5*c*((d-2*q*cd)*x32-d*q2*x53)
        for i in range(12):
            u[i] = u[i] + disl2/(2*np.pi)*du[i]
    
    if disl3 != 0:
        du[0]=-alp4*(sd/r+qy*cd)   -alp5*(z*y11-q2*z32)
        du[1]= alp4*2*xy*sd+d*x11 -alp5*c*(x11-q2*x32)
        du[2]= alp4*(y*x11+xy*cd)  +alp5*q*(c*et*x32+xi*z32)
        du[3]= alp4*xi/r3*sd+xi*q*y32*cd+alp5*xi*(3*c*et/r5-2*z32-z0)
        du[4]= alp4*2*y0*sd-d/r3 +alp5*c/r3*(1-3*q2/r2)
        du[5]=-alp4*yy0           -alp5*(c*et*qr-q*z0)
        du[6]= alp4*(q/r3+y0*sdcd)   +alp5*(z/r3*cd+c*d*qr-q*z0*sd)
        du[7]=-alp4*2*xi*ppy*sd-y*d*x32 +alp5*c*((y+2*q*sd)*x32-y*q2*x53)
        du[8]=-alp4*(xi*ppy*cd-x11+y*y*x32) +alp5*(c*((d+2*q*cd)*x32-y*et*q*x53)+xi*qqy)
        du[9]=  -et/r3+y0*cdcd -alp5*(z/r3*sd-c*y*qr-y0*sdsd+q*z0*cd)
        du[10]= alp4*2*xi*ppz*sd-x11+d*d*x32-alp5*c*((d-2*q*cd)*x32-d*q2*x53)
        du[11]= alp4*(xi*ppz*cd+y*d*x32)+alp5*(c*((y-2*q*sd)*x32+d*et*q*x53)+xi*qqz)
        for i in range(12):
            u[i] = u[i] + disl3/(2*np.pi)*du[i]

    return u