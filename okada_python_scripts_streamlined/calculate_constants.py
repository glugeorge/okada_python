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
    c2 = {}
    xi = np.where(np.abs(xi) < eps, 0, xi)
    et = np.where(np.abs(et) < eps, 0, et)
    q  = np.where(np.abs(q) < eps, 0, q)
    c2['xi2'] = xi * xi
    c2['et2'] = et * et
    c2['q2']  = q * q
    c2['r2']  = c2['xi2'] + c2['et2'] + c2['q2']
    c2['r']   = np.sqrt(c2['r2'])
    r0_condition = c2['r'] == 0
    c2['r3'] = np.where(r0_condition, 0, c2['r']*c2['r2'])
    c2['r5'] = np.where(r0_condition, 0, c2['r3']*c2['r2'])
    c2['y'] = np.where(r0_condition, 0, et*cd + q*sd)
    c2['d'] = np.where(r0_condition, 0, et*sd - q*cd)

    c2['tt'] = np.where((q == 0) | (r0_condition), 0, np.arctan(xi * et / (q * c2['r'])))

    # kxi conditions
    rxi = c2['r'] + xi
    c2['alx'] = np.where(r0_condition, 0, np.where(kxi == 1, -np.log(c2['r'] - xi), np.log(rxi)))
    c2['x11'] = np.where((kxi == 1) | (r0_condition), 0, 1 / (c2['r'] * rxi))
    c2['x32'] = np.where((kxi == 1) | (r0_condition), 0, (c2['r'] + rxi) * c2['x11'] * c2['x11'] / c2['r'])

    # ket conditions
    ret = c2['r'] + et
    c2['ale'] = np.where(r0_condition, 0, np.where(ket == 1, -np.log(c2['r'] - et), np.log(ret)))
    c2['y11'] = np.where((ket == 1) | (r0_condition), 0, 1 / (c2['r'] * ret))
    c2['y32'] = np.where((ket == 1) | (r0_condition), 0, (c2['r'] + ret) * c2['y11'] * c2['y11'] / c2['r'])
    
    c2['ey']= np.where(r0_condition, 0, sd/c2['r']-c2['y']*q/c2['r3'])
    c2['ez']= np.where(r0_condition, 0, cd/c2['r']+c2['d']*q/c2['r3'])
    c2['fy']= np.where(r0_condition, 0, c2['d']/c2['r3']+c2['xi2']*c2['y32']*sd)
    c2['fz']= np.where(r0_condition, 0, c2['y']/c2['r3']+c2['xi2']*c2['y32']*cd)
    c2['gy']= np.where(r0_condition, 0, 2*c2['x11']*sd-c2['y']*q*c2['x32'])
    c2['gz']= np.where(r0_condition, 0, 2*c2['x11']*cd+c2['d']*q*c2['x32'])
    c2['hy']= np.where(r0_condition, 0, c2['d']*q*c2['x32']+xi*q*c2['y32']*sd)
    c2['hz']= np.where(r0_condition, 0, c2['y']*q*c2['x32']+xi*q*c2['y32']*cd)

    return c2 