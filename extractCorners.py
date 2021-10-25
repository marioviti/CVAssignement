from scipy.ndimage.measurements import label
from scipy.ndimage import gaussian_filter as G
from scipy.ndimage import sobel as S
from skimage import transform

def HarrisCornerDetector(I, k=0.05, w=3):
    Ix,Iy = S(I,axis=0),S(I,axis=1)
    M = [ G(Ix**2,w), G(Iy**2,w), G(Ix*Iy,w) ]
    traceM = M[0] + M[1]
    detM = M[0]*M[1] - M[2]**2
    R = detM - k*(traceM)**2
    return R

def orderCorners(corners):
    ordered_corners = np.zeros([4,2])
    for c in corners:
        d = np.sign(c-corners)
        updwn = d[:,0].sum()
        lr = d[:,1].sum()
        if updwn < 0 and lr < 0:
            ordered_corners[0,:] = c
        if updwn < 0 and lr > 0:
            ordered_corners[3,:] = c
        if updwn > 0 and lr > 0:
            ordered_corners[2,:] = c
        if updwn > 0 and lr < 0:
            ordered_corners[1,:] = c
    return ordered_corners
    
def computePerspective(ordered_corners):

    x0,y0 = np.array([-3,-3])
    x1,y1 = np.array([3,-3])
    x2,y2 = np.array([3,3])
    x3,y3 = np.array([-3,3])

    u0,v0 = ordered_corners[0]
    u1,v1 = ordered_corners[1]
    u2,v2 = ordered_corners[2]
    u3,v3 = ordered_corners[3]

    A = [[x0, y0, 1, 0, 0, 0, -x0*u0, -y0*u0],
         [x1, y1, 1, 0, 0, 0, -x1*u1, -y1*u1],
         [x2, y2, 1, 0, 0, 0, -x2*u2, -y2*u2],
         [x3, y3, 1, 0, 0, 0, -x3*u3, -y3*u3],
         [0, 0, 0, x0, y0, 1, -x0*v0, -y0*v0],
         [0, 0, 0, x1, y1, 1, -x1*v1, -y1*v1],
         [0, 0, 0, x2, y2, 1, -x2*v2, -y2*v2],
         [0, 0, 0, x3, y3, 1, -x3*v3, -y3*v3]]

    b = [ u0,u1,u2,u3,v0,v1,v2,v3 ]

    x = np.linalg.solve(A, b)
    M = [ [x[0],x[1],x[2]],
          [x[3],x[4],x[5]],
          [x[6],x[7],1.] ]
    M = np.array(M)
    return M

def computeAffine(ordered_corners):

    x0,y0 = np.array([-3,-3])
    x1,y1 = np.array([3,-3])
    x2,y2 = np.array([3,3])
    x3,y3 = np.array([-3,3])

    u0,v0 = ordered_corners[0]
    u1,v1 = ordered_corners[1]
    u2,v2 = ordered_corners[2]

    A = [[x0, y0, 1, 0, 0, 0, ],
         [x1, y1, 1, 0, 0, 0, ],
         [x2, y2, 1, 0, 0, 0, ],
         [0, 0, 0, x0, y0, 1, ],
         [0, 0, 0, x1, y1, 1, ],
         [0, 0, 0, x2, y2, 1, ],
        ]

    b = [ u0,u1,u2,v0,v1,v2 ]

    x = np.linalg.solve(A, b)
    M = [ [x[0],x[1],x[2]],
          [x[3],x[4],x[5]],
          [0.,0.,1.] ]
    M = np.array(M)
    return M

def extractCorners(I, clusters):
    # extrapolated corners
    x0,y0 = np.array([-10,-10])
    x1,y1 = np.array([32,-10])
    x2,y2 = np.array([32,29])
    x3,y3 = np.array([-10,29])

    p0 = np.array([x0,y0,1])
    p1 = np.array([x1,y1,1])
    p2 = np.array([x2,y2,1])
    p3 = np.array([x3,y3,1])

    P = np.array([p0,p1,p2,p3])
    
    BBox = []
    
    for i in range(len(clusters)):
        # each cluste corresponds to a code we will extract the first one to provide an input to a decoder 
        col,row,width = clusters[i].round().astype(int)
        # by construction the center should be inside a blak square we will therfore extract 
        # the central connected component and run corner detection to detect 4 angles.
        sub_I = I[col-width//2:col+width//2, row-width//2:row+width//2]
        label_sub_I = label(1-sub_I)[0]
        label_center = label_sub_I[width//2, width//2]
        center_cc = (label_sub_I!=label_center).astype(np.float32)
        corner_sub_I = HarrisCornerDetector(center_cc)
        corner_candidates = np.argwhere(corner_sub_I>0.9)

        clustering = DBSCAN(eps=6, min_samples=4).fit(corner_candidates)
        cluster_sizes = np.array([ sum(clustering.labels_==c) for c in range(np.max(clustering.labels_)+1) ])
        size_inidices = np.argsort(cluster_sizes)
        corners = np.array([ corner_candidates[clustering.labels_==c].mean(axis=0) for c in range(np.max(clustering.labels_)+1) ])[size_inidices[::-1]][:4]
        corners = orderCorners(corners)
        M = computeAffine(corners)#*0.9 + computePerspective(corners)*0.1
        
        Pp = (M@P.T).T
        Pp/=Pp[:,-1:]
        Pp = Pp[:,:-1]
        
        Pp_I = Pp*1.
        Pp_I[:,0] += col-width//2
        Pp_I[:,1] += row-width//2
        
        BBox += [Pp_I]
        
        
    return BBox, corners, label_sub_I, center_cc, corner_sub_I, sub_I, M, Pp, Pp_I
