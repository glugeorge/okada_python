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

def DCCON1(x,y,d,c0):
    eps = 1e-6
    sd = c0['sd']
    cd = c0['cd']
    c1 = {}
    x = np.where(np.abs(x) < eps, 0, x)
    y = np.where(np.abs(y) < eps, 0, y)
    d = np.where(np.abs(d) < eps, 0, d)

    c1['p'] = y*cd+d*sd
    c1['q'] = y*sd-d*cd
    c1['s'] = c1['p']*sd+c1['s']*cd
    c1['t']= c1['p']*cd-c1['q']*sd
    c1['xy']=x*y
    c1['x2']=x*x
    c1['y2']=y*y
    c1['d2']=d*d
    c1['r2']=c1['x2']+c1['y2']+c1['d2']
    c1['r'] = np.sqrt(c1['r2'])
    r0_condition = c1['r'] == 0
    c1['r3'] = np.where(r0_condition, 0, c1['r']*c1['r2'])
    c1['r5'] = np.where(r0_condition, 0, c1['r3']*c1['r2'])
    c1['r7'] = np.where(r0_condition, 0, c1['r5']*c1['r2'])

    c1['a3'] = np.where(r0_condition, 0, 1-3*c1['x2']/c1['r2'])
    c1['a5'] = np.where(r0_condition, 0, 1-5*c1['x2']/c1['r2'])
    c1['b3'] = np.where(r0_condition, 0, 1-3*c1['y2']/c1['r2'])
    c1['c3'] = np.where(r0_condition, 0, 1-3*c1['d2']/c1['r2'])

    c1['qr'] = np.where(r0_condition, 0, 3*c1['q']/c1['r5'])
    c1['qrx'] = np.where(r0_condition, 0, 5*c1['qr']*x/c1['r2'])

    c1['uy'] = np.where(r0_condition, 0, sd - 5*y*c1['q']/c1['r2'])
    c1['uz'] = np.where(r0_condition, 0, cd + 5*d*c1['q']/c1['r2'])
    c1['vy'] = np.where(r0_condition, 0, c1['s'] - 5*y*c1['p']*c1['q']/c1['r2'])
    c1['vz'] = np.where(r0_condition, 0, c1['t'] + 5*d*c1['p']*c1['q']/c1['r2'])
    c1['wy'] = np.where(r0_condition, 0, c1['uy']+sd)
    c1['wz'] = np.where(r0_condition, 0, c1['uz']+cd)
    return c1


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