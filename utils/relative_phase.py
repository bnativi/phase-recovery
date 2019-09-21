from utils.heisenberg_weyl_methods import *

def get_QSa(aBin_var,m): #builds the matrix Q to be used to create the eigenbasis for Sa
    first = 0
    for i in range(m):         #records the index of the first 1 in a
        if (aBin_var[i] == 1):
            first = i
            break
        #end loop
    finalMatrix = []
    
    for i in range(m):   #build up each row of Q
        row = []         #Q is the identity matrix where the first row is shifted down into the "first" index,
                         #and the other rows are shifted up
                         # example: [0,1,0,0]
                         #          [0,0,1,0]
                         #          [1,0,0,0] if first = 2
                         #          [0,0,0,1] 
        
        if (i<first):   #first rows are shifted up
            for j in range(m):
                if (j == i+1):
                    row.append(1)
                else:
                    row.append(0)
                    
        elif (i==first): #the row which is [1,0,...,0]
            row.append(1)
            for j in range(m-1):
                row.append(0)

        else:            #remaining rows are the normal rows of the identity matrix
            for j in range(m):
                if (j == i):
                    row.append(1)
                else:
                    row.append(0)
        finalMatrix.append(row)

    return finalMatrix

def get_PSa(aBin_var,m): #builds the matrix P to be used to create the eigenbasis for Sa
    new = []            #copies aBin into new
    for i in range(m):  
        new.append(aBin_var[i])
        
    for i in range(m): #removes the first occuring 1 of new
        if (new[i] == 1):
            new[i] = 0
            break
            
    finalMatrix = []
    
    for i in range(m): #builds P where the first row is new and the first column is new.  All other entries are 0
        if (i == 0):
            finalMatrix.append(new)
        else:
            row = []
            for j in range(m):
                if (j==0):
                    row.append(new[i])
                else:
                    row.append(0)
            finalMatrix.append(row)
    
    return finalMatrix

def get_PTa(aBin_var,m): #builds the matrix P to be used to create the eigenbasis for Ta
    new = []            #copies aBin into new
    for i in range(m):  
        new.append(aBin_var[i])
        
    for i in range(m): #removes the first occuring 1 of new
        if (new[i] == 1):
            new[i] = 0
            break
            
    finalMatrix = []
    
    for i in range(m): #builds P where the first row is new and the first column is new.  All other entries are 0
        if (i == 0):
            row = []
            for j in range(m):
                if (j == 0):
                    row.append(1)
                else:
                    row.append(new[j])
            finalMatrix.append(row)
            
        else:
            row = []
            for j in range(m):
                if (j==0):
                    row.append(new[i])
                else:
                    row.append(0)
            finalMatrix.append(row)
    
    return finalMatrix

def diag_TP(P,m):        #computes the operator that corresponds to the matrix [[I, P], [0, I]]
                            #in this version of the method, the first input is P
    final = []              #will return a diagonal matrix with i^{vPv^T} for the diagonal entries
        # print(np.array(P)
    for i in range(2**m):   #loop over all v of length m
        v = int_to_bin(i,2**m)
        exp = np.matmul(v,np.matmul(P,np.transpose(v))) #compute vPv^T
        val = 1j ** exp

        row = []
        for j in range(2**m): #build corresponding row vector to add to the final matrix
            if (j == i):
                row.append(val) #i^{vRv^T} on the diagonal
            else:
                row.append(0)   #0 else

        final.append(row)

    return np.array(final)

def perm_Q(Q,m): #computes the operator that corresponds to the matrix [[Q, 0], [0, Q^T]] 
                    #in this version, the first input is the matrix Q
    final = []    #will return the corresponding operator
#     print(np.array(Q)
    for i in range(2**m):    #loops over all binary numbers of length m
        v = int_to_bin(i,2**m)
#         print("v", v
        vQ = np.matmul(v,Q)  #send v -> vQ
        for i in range(m):   #mod by 2 to get to binary
            vQ[i] = vQ[i] % 2
        #print(v, vQ)
        
        intvQ = bin_to_int(vQ)  #converts to integer
        evQ = []
        for i in range(2**m): #builds standard unit vector e_{vQ}
            if (i == intvQ):  #of length 2**m
                evQ.append(1)
            else:
                evQ.append(0)
        final.append(evQ)     #appends e_{vQ} to final matrix to build the permutation matrix corresponding to Q
    
    return np.transpose(np.array(final))

def get_Gk(k,m): #creates the operator gk that corresponds to the symplectic matrix Gk
    if (k==0):             #H_{2^0} x I_{2^m}
        return make_IN(m)
    elif (k==m):           #H_{2^m} x I_{2^0}
        return make_HN(m)
    else:                  #H_{2^k} x I_{2^(m-k)}
        H2k  = make_HN(k)
        I2mk = make_IN(m-k)
        return np.kron(H2k,I2mk)
    
def check(one, two): #checks if the two inputted vectors are the same up to a scalar multiple
    if(len(one) != len(two)):
        print("ERROR 1: check; len(one)!=len(two)", len(one), len(two))
        return
    nonzero = 0
    scalar = 0
    for i in range(len(one)):
        if (nonzero):
            if (one[i] != scalar * two[i]):
                print("fail", one[i], ":", scalar, two[i])
                return
        elif (one[i] != 0):
            scalar = two[i]/one[i]
            nonzero = 1
    if (scalar == 0):
        print("fail, scalar = 0")
    else:
        print("success with scalar =", scalar)
    return

# This function gets the bases for Sa and Ta given a choice of a and m
# Due to the testing in later blocks of code, this function appears to be working

def valid_a(a_var,m_var): #checks if the inputed a is valid.  If it is, it returns the binary version of a
    aBin = int_to_bin(a_var,2**m_var)
    if (a_var >= 2**m_var):
        print("ERROR 1: validA; a >= N for a =", a_var)
        return (0, [])
    elif( (np.dot(aBin,aBin) % 2) != 1):
        print("ERROR 2: validA; <a,a> = 0 for a =", a_var)
        return (0, [])
    return (1, aBin)  

def get_Sa_Ta_bases(a,m):
    (valid,aBin) = valid_a(a,m)
    if (not valid):
        print("ERROR 1: getSaTaBases; given a is not valid")
        return
    
    Q = get_QSa(aBin,m)
    PSa = get_PSa(aBin,m)
    PTa = get_PTa(aBin,m)

    tPSa = diag_TP(PSa,m)
    tPTa = diag_TP(PTa,m)
    g1 = get_Gk(1,m)
    HN  = make_HN(m)
    aQ = perm_Q(Q,m)

    gSa = np.matmul(np.matmul(np.matmul(tPSa,g1),HN),aQ) # gXNg^t = Sa
    basisSa = np.matmul(gSa.conj().T,HN)

    gTa = np.matmul(np.matmul(np.matmul(tPTa,g1),HN),aQ) # gXNg^t = Ta
    basisTa = np.matmul(gTa.conj().T,HN)
    
    return (basisSa,basisTa,gSa,gTa)

def symp_TP(P,m):
    final = []
    
    for i in range(m):
        row = []
        for j in range(m):
            if (j==i):
                row.append(1)
            else:
                row.append(0)
        for j in range(m):
            row.append(P[i][j])
        final.append(row)
        
    for i in range(m):
        row = []
        for j in range(m):
            row.append(0)
        for j in range(m):
            if (j==i):
                row.append(1)
            else:
                row.append(0)
        final.append(row)
    
    return np.array(final)

def symp_AQ(Q,m):
    final = []
    
    for i in range(m):
        row = []
        for j in range(m):
            row.append(Q[i][j])
        for j in range(m):
            row.append(0)
        final.append(row)
        
    for i in range(m):
        row = []
        for j in range(m):
            row.append(0)
        for j in range(m):
            row.append(Q[i][j])
        final.append(row)
    
    return np.array(final)

def symp_Gk(k,m): #builds the symplectic matrix Gk for given k and m
    final = []
    for i in range(m):              #example k=1, m=4
        row = []                    # [1,0,0,0,0,0,0,0]
        for j in range(2*m):        # [0,0,0,0,0,1,0,0]
            if (i<k):               # [0,0,0,0,0,0,1,0]
                if(j==i):           # [0,0,0,0,0,0,0,1]
                    row.append(1)   # [0,0,0,0,1,0,0,0]
                else:               # [0,1,0,0,0,0,0,0]
                    row.append(0)   # [0,0,1,0,0,0,0,0]
            else:                   # [0,0,0,1,0,0,0,0]
                if(j==m+i):
                    row.append(1)
                else:
                    row.append(0)
        final.append(row)
    
    for i in range(m):
        row = []
        for j in range(2*m):
            if (i<k):
                if(j==m+i):
                    row.append(1)
                else:
                    row.append(0)
            else:
                if(j==i):
                    row.append(1)
                else:
                    row.append(0)
        final.append(row)
                
    return np.array(final)

def get_GSa(a,m): #gets the symplect matrix G such that FG = (I | 0).  This matrix corresponds to the operator g that we found earlier.
    aBin = int_to_bin(a,2**m)   #we are finding this G because we need it to keep track of eigenvalues

    Q = get_QSa(aBin,m)
    P = get_PSa(aBin,m)

    TP = symp_TP(P,m)
    G1 = symp_Gk(1,m)
    AQ = symp_AQ(Q,m)
    
    return np.matmul(np.matmul(AQ,G1),TP)

def get_GTa(a,m): #gets the symplect matrix G such that FG = (I | 0).  This matrix corresponds to the operator g that we found earlier.
    aBin = int_to_bin(a,2**m)   #we are finding this G because we need it to keep track of eigenvalues

    Q = get_QSa(aBin,m)
    P = get_PTa(aBin,m)

    TP = symp_TP(P,m)
    G1 = symp_Gk(1,m)
    AQ = symp_AQ(Q,m)
    
    return np.matmul(np.matmul(AQ,G1),TP)

def get_c(aBin,bBin,m,G):   #finds the c such that (c,0) = (a,b)G
    vec = aBin + bBin

    newVec = np.matmul(vec,G)
    for i in range(len(newVec)):
        newVec[i] = newVec[i] %2
        
    return newVec[:m]

def extra(aBin,bBin):    #used to compute the extra factor of (-1) that is needed for the eigenvalues to match up with what they should be
    val = np.matmul(aBin,np.transpose(bBin))   #ideally, we don't need this function but I will use it until I realize what is wrong with my program
    val = val %4
    if (val == 1 or val == 2):
        return -1
    else:
        return 1  

def build_eigenvalue_matrix(a,m):    #builds the matrix of eigenvalues
    (valid,aBin) = valid_a(a,m)
#     print(a, aBin
    if (not valid):
        print("Error 1: buildEigvalMat; not valid")
        return []
    
    final = []
    
    # (basisSa,basisTa,gSa,gTa) = get_Sa_Ta_bases(a,m)
    GSa = get_GSa(a,m)
    GTa = get_GTa(a,m)
    
    for bBin1 in create_Omega_a(aBin,2**m):
        row = []
        
        cBin1 = get_c(aBin,bBin1,m,GSa) 
        e1 = E(aBin,bBin1)
        for i in range(2**(m)):
            vBin1 = int_to_bin(i,2**m)
            row.append(  extra(aBin,bBin1) * (-1)**(np.matmul(cBin1,np.transpose(vBin1))) * (1j)**(-np.matmul(aBin,bBin1))  )
        
        for i in range(2**(m)):
            row.append(0)
        
        final.append(row)
        
    for bBin2 in create_NOmega_a(aBin,2**m):
        row = []
        
        for i in range(2**(m)):
            row.append(0)
        
        cBin2 = get_c(aBin,bBin2,m,GTa) 
        e1 = E(aBin,bBin2)
        for i in range(2**m):
            vBin2 = int_to_bin(i,2**m)
            row.append(  extra(aBin,bBin2) * (-1)**(np.matmul(cBin2,np.transpose(vBin2))) * (1j)**(-np.matmul(aBin,bBin2))  )
            
        final.append(row)
    #print(np.array(final))
    return np.array(final)

def build_eigenvectors(x,a,m):  #builds the vector of intensity measurements with eigenvectors
    final = []
    x = np.array(x)
    
    (basisSa,basisTa,gSa,gTa) = get_Sa_Ta_bases(a,m)
    
    for h in np.transpose(basisSa):
        mat = np.matmul( np.outer(x,x.conj()) , np.outer(h,h.conj()) ) #xx^thh^t   w/ conj   x,h
        final.append( np.trace(mat) )                                  #|<x,h>|^2 = Tr(xx^thh^t)
        
    for f in np.transpose(basisTa):
        mat = np.matmul( np.outer(x,x.conj()) , np.outer(f,f.conj()) ) #xx^tff^t   w/ conj   x,f
        final.append( np.trace(mat) )                                   #|<x,f>|^2 = Tr(xx^tff^t)
    
    return np.transpose(np.array(final))

def relative_phase(x,a,m): #returns the relative phases of a vector x for a given choice of a
    
    HN = make_HN(m)
    Emat = build_eigenvalue_matrix(a,m)
    mVec = build_eigenvectors(x,a,m)
#     print
#     print("HN")
#     print(HN)
#     print("matrix of eigenvalues")
#     print(Emat)
#     print("matrix of inner products")
#     print(mVec)
    return (1/np.sqrt(2**m)) * np.matmul(np.matmul(HN,Emat),mVec)