def moebius(A, z):
    return (A[0, 0] * z + A[0, 1]) / (A[1, 0] * z + A[1, 1])
