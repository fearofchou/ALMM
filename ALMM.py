import numpy as np
import collections as cl
import multiprocessing as mp
import sys
import time
#sys.path.append('/home/fearofchou/ND/m195/max/paper/MM16/code/eval')
#import evalu
import pickle

class PF():
    def __init__(self, ls=10, reg=1, alp=1, iters=10):
        print 'Model parameter'
        self.cc = mp.cpu_count()
        self.cc = 5
        self.ls = ls
        self.reg = reg
        self.alp = alp
        self.iters = iters
        self.mn = 'PF'
        self.output = '%s iters=%d factor=%d alpha=%d reg=%f' % (
            self.mn, self.iters, self.ls, self.alp, self.reg)

        print self.output

    def _init_f(self, uidx, tidx, nidx):
        self.uLF = {}
        for i in uidx:
            self.uLF[i] = np.random.normal(size=(self.ls))
        self.tLF = {}
        for i in tidx:
            self.tLF[i] = np.random.normal(size=(self.ls))
        self.nLF = {}
        for i in nidx:
            self.nLF[i] = np.random.normal(size=(self.ls))

    def _init_w(self, fs):
        self.wt = np.random.normal(size=(fs,self.ls))
        self.wn = np.random.normal(size=(fs,self.ls))


def evl_RMSE(i):
    u,t,e,p = R[i]
    prd = np.dot(m.uLF[u], m.tLF[t])+ np.dot(m.uLF[u], m.nLF[e]) + np.dot(m.tLF[t], m.nLF[e])
    RMSE = (p - prd)**2
    return RMSE


# update function ((X+Y)(X+Y)T + LU)-1 * (C-XY)(X+Y)
def update_U(u):
    C = 1 + m.alp * np.log(1 + R[uidx[u],3])
    X = np.array([m.tLF[i] for i in R[uidx[u],1]])
    Y = np.array([m.nLF[i] for i in R[uidx[u],2]])
    I = m.reg*np.eye(m.ls)

    XY = X+Y
    XYXY = XY.T.dot(XY)
    C_XY = C
    for idx,i in enumerate(C):
        C_XY[idx] = C_XY[idx] - np.dot(X[idx],Y[idx])
    C_XYXY = np.dot(XY.T,C_XY)
    return [u,  np.dot(np.linalg.inv(XYXY+I), C_XYXY)]

def update_T(t):
    C = 1 + m.alp * np.log(1 + R[tidx[t],3])
    X = np.array([m.uLF[i] for i in R[tidx[t],0]])
    Y = np.array([m.nLF[i] for i in R[tidx[t],2]])
    I = m.reg*np.eye(m.ls)

    XY = X+Y
    XYXY = XY.T.dot(XY)
    C_XY = C
    for idx,i in enumerate(C):
        C_XY[idx] = C_XY[idx] - np.dot(X[idx],Y[idx])
    C_XYXY = np.dot(XY.T,C_XY)
    return [t,  np.dot(np.linalg.inv(XYXY+I), C_XYXY)]

def update_N(n):
    C = 1 + m.alp * np.log(1 + R[nidx[n],3])
    X = np.array([m.uLF[i] for i in R[nidx[n],0]])
    Y = np.array([m.tLF[i] for i in R[nidx[n],1]])
    I = m.reg*np.eye(m.ls)

    XY = X+Y
    XYXY = XY.T.dot(XY)
    C_XY = C
    for idx,i in enumerate(C):
        C_XY[idx] = C_XY[idx] - np.dot(X[idx],Y[idx])
    C_XYXY = np.dot(XY.T,C_XY)
    return [n,  np.dot(np.linalg.inv(XYXY+I), C_XYXY)]

def fea_w(t, sid, reg):
    A = np.array([F[i] for i in sid])
    I = reg*np.eye(fs)

    if t == 'tLF':
        X = np.array([m.tLF[i] for i in sid])
    if t == 'nLF':
        X = np.array([m.nLF[i] for i in sid])

    AA = np.linalg.inv(np.dot(A.T,A)+I)

    if t == 'tLF':
        m.wt  =np.dot( np.dot(X.T,A), AA)
    if t == 'nLF':
        m.wn  =np.dot( np.dot(X.T,A), AA)

def put_LF(utn):
    if utn == 'u':
        for u, f in LF:
            m.uLF[u] = f
    if utn == 't':
        for t, f in LF:
            m.tLF[t] = f
    if utn == 'n':
        for n, f in LF:
            m.nLF[n] = f

def load_data():
    R    = np.genfromtxt('./Rating.dat',dtype=int,delimiter=',',skip_header=1)
    fea  = np.genfromtxt('./feature.dat',dtype=float,delimiter=',',skip_header=1)

    F = {}
    for i in xrange(len(fea)):
        F[int(fea[i][0])+1] = fea[i][1:]

    uidx = cl.defaultdict(list)
    tidx = cl.defaultdict(list)
    nidx = cl.defaultdict(list)

    for idx,val in enumerate(R):
        u,t,n,p = val
        uidx[u].append(idx)
        tidx[t].append(idx)
        nidx[n].append(idx)

    fs = len(fea[i][1:])
    return R, F, uidx, tidx, nidx, fs

# MAIN
#load_data
R, F, uidx, tidx, nidx, fs = load_data()
# init model
m = PF()
m._init_f(uidx.keys(), tidx.keys(), nidx.keys())
m._init_w(fs)
w_reg = 0.1
print 'Training model'
for i in range(m.iters):
    mt = time.time()

    #update user latent factor
    mup = mp.Pool(processes=m.cc)
    LF = mup.map(update_U, uidx.keys())
    mup.close()
    mup.join()
    put_LF('u')

    #update this-item latent factor
    mup = mp.Pool(processes=m.cc)
    LF = mup.map(update_T, tidx.keys())
    mup.close()
    mup.join()
    put_LF('t')

    #update the weight between this item latent factor and content feature
    fea_w('tLF', tidx.keys(), w_reg )
    #Mapping this item factor into content space
    for j in m.tLF:
        m.tLF[j] = np.dot(m.wt, F[j])

    #update next-item latent factor
    mup = mp.Pool(processes=m.cc)
    LF = mup.map(update_N, nidx.keys())
    mup.close()
    mup.join()
    put_LF('n')

    #update the weight between next item latent factor and content feature
    fea_w('nLF', nidx.keys(), w_reg )
    #Mapping this item factor into content space
    for j in m.nLF:
        m.nLF[j] = np.dot(m.wn, F[j])

    # calulate training RMSE
    et = time.time()
    mup = mp.Pool(processes=48)
    RE = np.array(mup.map(evl_RMSE, xrange(len(R))))
    mup.close()
    mup.join()
    print 'Train %2d loss=%f time=%f' % (i+1, np.sqrt(RE.mean()),time.time()-mt)


