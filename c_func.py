#from numba import njit
from math import cos, sin, sqrt

#@njit
def casadi_sq(x):
    return x*x
    
#@njit
def casadi_f0(arg):
    res = [[0] * 24]
    
    a0 = 5.4000000000000003e-03
    a1 = arg[0][0] if arg[0].size > 0 else 0
    a2 = sin(a1)
    a3 = (a0 * a2)
    a4 = (-a3)
    res[0][0] = a4
    a4 = cos(a1)
    a0 = (a0 * a4)
    a5 = (-a0)
    res[0][1] = a5
    a5 = 2.8480000000000000e-01
    res[0][2] = a5
    a6 = -6.4000000000000003e-03
    a7 = (a6 * a2)
    a8 = -2.1040000000000000e-01
    a9 = cos(a1)
    a10 = arg[0][1] if arg[0].size > 1 else 0
    a11 = sin(a10)
    a12 = (a9 * a11)
    a13 = (a8 * a12)
    a7 = (a7 - a13)
    a7 = (a7 - a3)
    res[0][3] = a7
    a1 = sin(a1)
    a11 = (a1 * a11)
    a3 = (a8 * a11)
    a13 = (a6 * a4)
    a3 = (a3 + a13)
    a3 = (a3 - a0)
    res[0][4] = a3
    a0 = cos(a10)
    a13 = (a8 * a0)
    a5 = (a5 - a13)
    res[0][5] = a5
    a13 = 6.4000000000000003e-03
    a14 = cos(a10)
    a9 = (a9 * a14)
    a15 = arg[0][2] if arg[0].size > 2 else 0
    a16 = sin(a15)
    a17 = (a9 * a16)
    a18 = cos(a15)
    a19 = (a2 * a18)
    a17 = (a17 + a19)
    a19 = (a13 * a17)
    a20 = (a8 * a12)
    a19 = (a19 + a20)
    a7 = (a7 - a19)
    res[0][6] = a7
    a1 = (a1 * a14)
    a14 = (a1 * a16)
    a18 = (a4 * a18)
    a14 = (a14 - a18)
    a18 = (a13 * a14)
    a19 = (a8 * a11)
    a18 = (a18 + a19)
    a18 = (a18 + a3)
    res[0][7] = a18
    a10 = sin(a10)
    a16 = (a10 * a16)
    a13 = (a13 * a16)
    a8 = (a8 * a0)
    a13 = (a13 - a8)
    a13 = (a13 + a5)
    res[0][8] = a13
    a5 = (a6 * a17)
    a8 = -2.0840000000000000e-01
    a3 = cos(a15)
    a9 = (a9 * a3)
    a15 = sin(a15)
    a2 = (a2 * a15)
    a9 = (a9 - a2)
    a2 = arg[0][3] if arg[0].size > 3 else 0
    a19 = sin(a2)
    a20 = (a9 * a19)
    a21 = cos(a2)
    a22 = (a12 * a21)
    a20 = (a20 + a22)
    a22 = (a8 * a20)
    a5 = (a5 - a22)
    a5 = (a5 + a7)
    res[0][9] = a5
    a1 = (a1 * a3)
    a4 = (a4 * a15)
    a1 = (a1 + a4)
    a4 = (a1 * a19)
    a15 = (a11 * a21)
    a4 = (a4 + a15)
    a15 = (a8 * a4)
    a7 = (a6 * a14)
    a15 = (a15 - a7)
    a15 = (a15 + a18)
    res[0][10] = a15
    a10 = (a10 * a3)
    a19 = (a10 * a19)
    a21 = (a0 * a21)
    a19 = (a19 - a21)
    a8 = (a8 * a19)
    a6 = (a6 * a16)
    a8 = (a8 - a6)
    a8 = (a8 + a13)
    res[0][11] = a8
    a13 = -1.0589999999999999e-01
    a6 = (a13 * a20)
    a5 = (a5 - a6)
    res[0][12] = a5
    a6 = (a13 * a4)
    a6 = (a6 + a15)
    res[0][13] = a6
    a15 = (a13 * a19)
    a15 = (a15 + a8)
    res[0][14] = a15
    a8 = cos(a2)
    a9 = (a9 * a8)
    a2 = sin(a2)
    a12 = (a12 * a2)
    a9 = (a9 - a12)
    a12 = arg[0][4] if arg[0].size > 4 else 0
    a21 = cos(a12)
    a9 = (a9 * a21)
    a12 = sin(a12)
    a17 = (a17 * a12)
    a9 = (a9 - a17)
    a17 = arg[0][5] if arg[0].size > 5 else 0
    a3 = sin(a17)
    a9 = (a9 * a3)
    a17 = cos(a17)
    a20 = (a20 * a17)
    a9 = (a9 + a20)
    a20 = (a13 * a9)
    a5 = (a5 - a20)
    res[0][15] = a5
    a4 = (a4 * a17)
    a11 = (a11 * a2)
    a1 = (a1 * a8)
    a11 = (a11 - a1)
    a11 = (a11 * a21)
    a14 = (a14 * a12)
    a11 = (a11 + a14)
    a11 = (a11 * a3)
    a4 = (a4 - a11)
    a11 = (a13 * a4)
    a11 = (a11 + a6)
    res[0][16] = a11
    a19 = (a19 * a17)
    a16 = (a16 * a12)
    a10 = (a10 * a8)
    a0 = (a0 * a2)
    a10 = (a10 + a0)
    a10 = (a10 * a21)
    a16 = (a16 - a10)
    a16 = (a16 * a3)
    a19 = (a19 - a16)
    a13 = (a13 * a19)
    a13 = (a13 + a15)
    res[0][17] = a13
    a15 = -1.3150000000000001e-01
    a16 = (a15 * a9)
    a16 = (a5 - a16)
    res[0][18] = a16
    a16 = (a15 * a4)
    a16 = (a16 + a11)
    res[0][19] = a16
    a15 = (a15 * a19)
    a15 = (a15 + a13)
    res[0][20] = a15
    a15 = 1.2000000000000000e-01
    a16 = (a15 * a9)
    a3 = -6.1499999999999999e-02
    a9 = (a3 * a9)
    a5 = (a5 - a9)
    a16 = (a16 + a5)
    res[0][21] = a16
    a16 = (a3 * a4)
    a16 = (a16 + a11)
    a4 = (a15 * a4)
    a16 = (a16 - a4)
    res[0][22] = a16
    a3 = (a3 * a19)
    a3 = (a3 + a13)
    a15 = (a15 * a19)
    a3 = (a3 - a15)
    res[0][23] = a3
    
    return res


# a = np.array([0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)

# b = casadi_f0(a)

# c = []

# print(b, "\n")
# print(len(b))
# print(np.shape(b))

# print("****************************")
# print(np.shape(b[0]))
# print("****************************")
# print("****************************")

# print(b[0][21])
# print(b[0][22])
# print(b[0][23])