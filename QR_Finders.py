import numpy as np

def QR_Finders(I, tol=0.5):
    H,W = I.shape
    I[:,0] = 0
    I[:,-1] = 0
    candidates = []
    for i in range(0,H,4):
        Ix_i =  I[i,:][1:] - I[i,:][:-1]
        up_j = np.argwhere(Ix_i==1)
        if len(up_j) == 1: continue
        up_j = up_j.squeeze()
        dwn_j = np.argwhere(Ix_i==-1).squeeze()
        w = dwn_j-up_j
        b = up_j[1:] - dwn_j[:-1]
        starts = dwn_j[:-1]
        ends = up_j[1:]
        for j in range(1,len(b)-1):
            Finder_j = 3.0-tol < b[j]/w[j+1] < 3.0 +tol
            Finder_j = 3.0-tol < b[j]/w[j] < 3.0 + tol and Finder_j
            Finder_j = 1.-tol< b[j-1]/w[j] < 1.+tol and Finder_j
            Finder_j = 1.-tol< b[j+1]/w[j+1] < 1.+tol and Finder_j
            if Finder_j:
                A = starts[j-1]
                B = ends[j+1]
                candidates += [ (i,A,B) ]
    return candidates
