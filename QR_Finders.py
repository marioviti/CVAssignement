import numpy as np

def QR_Finders(I, tol=0.5, min_width=20):
    H,W = I.shape
    I[:,0] = 0
    I[:,-1] = 0
    candidates = [ [] for i in range(H) ] 
    for i in range(0,H):
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
            if b[j] < min_width: continue
            is_finder = 3.0-tol < b[j]/w[j+1] < 3.0 +tol
            is_finder = 3.0-tol < b[j]/w[j] < 3.0 + tol and is_finder
            is_finder = 1.-tol< b[j-1]/w[j] < 1.+tol and is_finder
            is_finder = 1.-tol< b[j+1]/w[j+1] < 1.+tol and is_finder
            if is_finder:
                A = starts[j-1]
                B = ends[j+1]
                candidates[i] += [ [A,B] ]
    return candidates
