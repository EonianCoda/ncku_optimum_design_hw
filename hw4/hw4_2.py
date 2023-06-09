
def cal_F(M: list, 
          A: list, 
          D: list, 
          n: float):
        A1, A2, A3, A4, A5, A6, A7, A8, A9 = A
        d1, d2, d3a, d3k, d4, d5k, d5h, d6, d7k, d7h, d8, d9 = D 
        M1, M2, M3 = M
        c11 = ((d1 * d1) * (A1 * A1)) + ((d2 * d2) * (A2 * A2)) + ((d3a * d3a) * (A3 * A3))
        c12 = (d3a * d3k * (A3 * A3))
        c22 = ((d3k * d3k) * (A3 * A3)) + ((d4 * d4) * (A4 * A4)) + ((d5k * d5k) * (A5 * A5)) + ((d6 * d6) * (A6 * A6)) + ((d7k * d7k) * (A7 * A7))
        c23 = ((d5k * d5h * (A5 * A5))) + (d7k * d7h * (A7 * A7))
        c33 = ((d5h * d5h) * (A5 * A5)) + ((d7h * d7h) * (A7 * A7)) + ((d8 * d8) * (A8 * A8)) + ((d9 * d9) * (A9 * A9))
        
        denominator = ((c11 * c22 * c33) - ((c12 * c12) * c33) - ((c23 * c23) * c11))
        lam1 = (2 * ((M1 * c33 * c22) - (M1 * (c23 * c23)) - (M2 * c12 * c33) + (M3 * c12 * c23))) / denominator
        lam2 = (2 * ((-M1 * c12 * c33) + (M2 * c11 * c33) - (M3 * c23 * c11))) / denominator
        lam3 = (2 * ((M1 * c12 * c23) - (M2 * c11 * c23) + (M3 * c11 * c22) - (M3 * (c12 * c12)))) / denominator
    
        F1 = (lam1 * d1 * (A1 ** n)) / n
        F2 = -(lam1 * d2 * (A2 ** n)) / 2
        F3 = -((lam1 * d3a + lam2 * d3k) * (A3 ** n)) / n
        F4 = (lam2 * d4 * (A4 ** n)) / n
        F5 = ((lam2 * d5k + lam3 * d5h) * (A5 ** n)) / n
        F6 = -(lam2 * d6 * (A6 ** n)) / n
        F7 = -((lam2 * d7k + lam3 * d7h) * (A7 ** n)) / n
        F8 = (lam3 * d8 * (A8 ** n)) / n
        F9 = -(lam3 * d9 * (A9 ** n)) / n
        return F1, F2, F3, F4, F5, F6, F7, F8, F9

if __name__ == '__main__':
    n = 2
    D = [0.0298, 0.044, 0.044, 0.0138, 0.0329, 0.0329, 0.0279, 0.025, 0.025, 0.0619, 0.0317, 0.0368]
    A = [11.5, 92.5, 44.3, 98.1, 20.1, 6.1, 45.5, 31.0, 44.3]
    M = [4, 33, 31]
    D_indices = [[0], [1], [2,3], [4], [5,6], [7], [8,9], [10], [11]]

    F = cal_F(M, A, D, n)
    # print(F)
    for i, f in enumerate(F):
        if f < 0:
            for d_idx in D_indices[i]:
                D[d_idx] = 0
    F = cal_F(M, A, D, n)
    F = [abs(f) for f in F]
    for i in range(len(F)):
        print('F{} = {:.1f}'.format(i + 1, F[i]))