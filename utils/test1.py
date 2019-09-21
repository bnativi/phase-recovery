def clean_up_matrix(matrix): #code that rounds entries of matrices so that testing my code is easier.  Some errors occur when the entries of the matrices are irrational    
    final = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[i])):
            row.append(round(matrix[i][j].real,5)+round(matrix[i][j].imag,5)*1j + 0j)
        final.append(row)
    return np.array(final)
                        
                      #ASSUMES ALL ENTRIES ARE REAL (NOT COMPLEX)
def clean_up_matrix_2(matrix): #code that rounds entries of matrices so that testing my code is easier.  Some errors occur when the entries of the matrices are irrational    
    final = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[i])):
            row.append(round(matrix[i][j].real,5))
        final.append(row)
    return np.array(final)

def round_complex(complex_num): #rounds a the real and complex components of a complex number to 5 decimal places
    return round(complex_num.real,5)+round(complex_num.imag,5)*1j + 0j

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

def check_3(vec1,vec2): #check function to show that two complex valued vectors are equal.
    if (len(vec1) != len(vec2)):
        print("FAIL lengths not equal:", len(vec1), len(vec2))
        return False
    for i in range(len(vec1)):
        if (round_complex(vec1[i]) != round_complex(vec2[i])):
            print("FAIL not equal at", i, ":", round_complex(vec1[i]), round_complex(vec2[i]))
            return False
    print("SUCCESS")
    return True