import numpy as np
import math
import scipy.misc
import scipy.special


def em(m1, m2):
    value = 0.0
    if m1 == 0:
        value = 1
    else:
        value = np.sign(m1)
    if m2 == 0:
        value = value * 1
    else:
        value = value * np.sign(m2)
        # print("em {} {} {: .20E}".format(m1, m2, value))
    return value


def clm(l1, l2, L, m1, m2, M):
    if M == m1 + m2:
        kd = 1
    else:
        kd = 0
    value = 0.0
    if L >= M:
        tvalue = []
        lower = max(0, (l1 - m1 - (L - M)), (l2 + m2 - (L + M)))
        upper = min((l1 - m1), (l2 + m2), (l1 + l2 - L))
        for t in range(lower, upper + 1):
            interim = ((-1) ** t) * scipy.special.binom(l1 + l2 - L, t)
            interim = interim * scipy.special.binom(L - M, l1 - m1 - t)
            interim = interim * scipy.special.binom(L + M, l2 + m2 - t)
            tvalue.append(interim)
        interim = (2 * l1 + 1) * (2 * l2 + 1) * scipy.special.binom(l1 + l2 + L + 1, l1 - l2 + L)
        interim = interim * scipy.special.binom(l1 + l2 + L + 1, l2 - l1 + L)
        interim = interim * scipy.special.binom(2 * l1, l1 + m1) * scipy.special.binom(2 * l2, l2 + m2)
        interim = 1.0 / interim
        interim = np.sqrt(interim * (2 * L + 1) * (2 * L + 1) * scipy.special.binom(l1 + l2 + L + 1,
                                                                                    l1 + l2 - L) * scipy.special.binom(
            2 * L, L + M))
        value = interim * math.fsum(tvalue)
        interim = (m1 + abs(m1) + m2 + abs(m2) + M + abs(M)) / 2
        interim = (-1) ** interim
        value = value * interim * kd
    else:
        value = 0.0
        # print("clm {} {} {} {} {} {}: {: .20E}".format(l1, l2, L, m1, m2, M, value))
    return value


def tlm(a, l1, m1, l2, m2, theta, phi):
    value = 0.0
    y1 = abs(m1)
    y2 = abs(m2)
    if a == 0:
        kd1 = 1
    else:
        kd1 = 0
    if m1 == 0:
        kd2 = 1
    else:
        kd2 = 0
    if m2 == 0:
        kd3 = 1
    else:
        kd3 = 0
    if y1 == y2 and em(m1, m2) == -1:
        value = 0.0
    else:
        ivalue = []
        for i in [-1, 1]:
            Lvalue = []
            for L in range(abs(l1 - l2), l1 + l2 + 1, 2):
                if i == int(em(m1, m2)):
                    kd4 = 1
                else:
                    kd4 = 0
                if int(em(m1, m2) * abs(i * y1 + y2)) == 0:
                    kd5 = 1
                else:
                    kd5 = 0
                    # in1 = em(m1, 0) ** (kd4)
                in1 = em(i * y1 + y2, m1) ** (kd4)
                in2 = clm(l1, l2, L, i * y1, y2, i * y1 + y2)
                in3 = clm(l1, l2, L, a, -a, 0)
                in4 = np.sqrt(((2 * math.pi) / (2 * L + 1)) * (1 + kd5))
                in5 = slm(L, em(m1, m2) * abs(y2 + i * y1), theta, phi)
                Lvalue.append(in1 * in2 * in3 * in4 * in5)
            ivalue.append(math.fsum(Lvalue))
        value = math.fsum(ivalue) * (2 * ((-1) ** (y1 + y2))) / ((1 + kd1) * np.sqrt((1 + kd2) * (1 + kd3)))
        # print("tlm {} {} {} {} {} {} {}: {: .20E}".format(a, l1, m1, l2, m2, theta, phi, value))
    return value


def slm(l1, m1, theta, phi):
    value = 0.0
    if abs(m1) > l1:
        value = 0.0
    else:
        value = plm(l1, abs(m1), theta) * Phi(m1, phi)
        # print("slm {} {} {} {}: {: .20E}".format(l1, m1, theta, phi, value))
    return value


def Phi(m1, phi):
    if m1 == 0:
        kd = 1
    else:
        kd = 0
    value = (1) / (np.sqrt(math.pi * (1 + kd)))
    if m1 >= 0:
        value = value * np.cos(abs(m1) * phi)
    else:
        value = value * np.sin(abs(m1) * phi)
        # print("Phi {} {}: {: .20E}".format(m1, phi, value))
    return value


def plm(l, a, theta):
    value = []
    limit = (l - a - ((1 - ((-1) ** (l - a))) / (2))) / 2
    for k in range(0, int(limit) + 1):
        interim = ((-1) ** k) * scipy.special.binom(a + k, k) * scipy.special.binom(2 * l - 2 * k, l - k)
        interim = interim * scipy.special.binom(l - k, l - a - 2 * k) * (np.cos(theta) ** (l - a - 2 * k))
        value.append(interim)
    interim = ((((-1) ** a) * (np.sin(theta) ** a)) / (2 ** l))
    interim = interim * np.sqrt((2 * l + 1) / (2 * scipy.special.binom(l, a) * scipy.special.binom(l + a, a)))
    value = interim * math.fsum(value)
    # print("plm {} {} {}: {: .20E}".format(l, a, theta, value))
    return value


def An(k, p):
    value = []
    for j in range(0, k + 1):
        value.append((p ** j) / scipy.misc.factorial(j))
    value = math.fsum(value) * np.exp(-p) * ((scipy.misc.factorial(k)) / (p ** (k + 1)))
    # print("An ",k, p, " :", value)
    return value


def An3(k1, k, p):
    if p == 0.0:
        if k1 == (k + 1):
            value = scipy.misc.factorial(k)
        else:
            value = 0.0
    else:
        value = []
        for j in range(k1 - k - 1, k1 - 1 + 1):
            value.append((p ** j) / (scipy.misc.factorial(j - k1 + k + 1)))
        value = math.fsum(value) * scipy.misc.factorial(k) * np.exp(-p)
        # print("An3 ", k1, k, p, " :", value)
    return value


def Bn3(k, p):
    value = []
    if p == 0.0:
        value = (((-1) ** k) + 1) / (1 + k)
    elif p < 2.0:
        if k % 2 == 0:
            for i in range(0, 202, 2):
                value.append((p ** i) / (scipy.misc.factorial(i)) / (i + k + 1))
            value = math.fsum(value) * 2.0
        else:
            for i in range(1, 203, 2):
                value.append((p ** i) / (scipy.misc.factorial(i)) / (i + k + 1))
            value = math.fsum(value) * -2.0
    else:
        value = (((-1) ** (k + 1)) * An(k, -p)) - An(k, p)
        # print("Bn3 ", k, p, " :", value)
    return value


def Fmn(m, n1, n2):
    value = []
    for sigma in range(int(((m - n1) + abs(m - n1)) / 2), min(m, n2) + 1):
        value.append(((-1) ** sigma) * scipy.special.binom(n1, m - sigma) * scipy.special.binom(n2, sigma))
        # print("Fmn ",m ,n1, n2, " :", math.fsum(value))
    return math.fsum(value)


def dlbt(l, lambda_, beta):
    value = ((-1) ** (((l - beta)) / 2)) / (2 ** l)
    value = value * np.sqrt((((2 * l + 1) / 2) * scipy.special.binom(l + lambda_, l)) / scipy.special.binom(l, lambda_))
    value = value * scipy.special.binom(l, (l - beta) / 2) * scipy.special.binom(l + beta, beta - lambda_)
    # print("dlbt ",l, lambda_, beta, " :", value)
    return value


def galbet(l1, l2, lambda_, alpha, beta):
    value = []
    for i in range(0, lambda_ + 1):
        value.append(((-1) ** i) * scipy.special.binom(lambda_, i) * dlbt(l1, lambda_, alpha + 2 * lambda_ - 2 * i))
    value = math.fsum(value) * dlbt(l2, lambda_, beta)
    # print("galbet ",l1, lambda_, l2, lambda_, alpha, beta, " :", value)
    return value


def overlap(n1, l1, n2, l2, lambda_, p, t):
    p1 = 1 + t
    p2 = 1 - t
    pt = p * t
    if (l1 - lambda_) % 2 == 0:
        a1 = -lambda_
    else:
        a1 = -lambda_ + 1
    if (l2 + lambda_) % 2 == 0:
        b1 = lambda_
    else:
        b1 = lambda_ + 1
    numerator = ((p1 ** (n1 + 0.5)) * ((p2 ** (n2 + 0.5))))
    denominator = np.sqrt(scipy.misc.factorial(2 * n1)) * np.sqrt(scipy.misc.factorial(2 * n2))
    b = numerator / denominator
    ivalue = []
    for i in range(a1, l1 + 1, 2):
        jvalue = []
        for j in range(b1, l2 + 1, 2):
            kvalue = []
            for k in range(0, i + j + 1):
                mvalue = []
                for m in range(0, n1 + n2 - i - j + 1):
                    mvalue.append(
                        Fmn(m, n1 - i, n2 - j) * Bn3(m + k, pt) * An3(n1 + n2 + 1, n1 + n2 - i - j - m + k, p))
                kvalue.append(math.fsum(mvalue) * Fmn(k, i + lambda_, j - lambda_))
            jvalue.append(math.fsum(kvalue) * galbet(l1, l2, lambda_, i, j))
        ivalue.append(math.fsum(jvalue))
    ivalue = math.fsum(ivalue) * b * ((-1) ** (l2 + lambda_))
    # print("overlap ",n1, l1, n2, l2, lambda_, p, t, " :", ivalue)
    return ivalue


def SlaterOverlap(n1, l1, m1, zeta1, n2, l2, m2, zeta2, r, theta, phi):
    p = (r / 2.0) * (zeta1 + zeta2)
    t = (zeta1 - zeta2) / (zeta1 + zeta2)
    S = []
    for lambda_ in range(0, min(l1, l2) + 1):
        S.append(tlm(lambda_, l1, m1, l2, m2, theta, phi) * overlap(n1, l1, n2, l2, lambda_, p, t))
        # print(S)
    return math.fsum(S)


def SlaterOverlapCartesian(n1, l1, m1, zeta1, x1, y1, z1, n2, l2, m2, zeta2, x2, y2, z2):
    x = (x2 - x1) * 1.889725989
    y = (y2 - y1) * 1.889725989
    z = (z2 - z1) * 1.889725989
    xy = x**2 + y**2
    r = np.sqrt(xy + z**2)
    theta = np.arctan2(np.sqrt(xy), z) # for elevation angle defined from Z-axis down
    #theta = np.arctan2(z, np.sqrt(xy)) # for elevation angle defined from XY-plane up
    phi = np.arctan2(y, x)

    # Old non-working code for Cartesian to Spherical conversion: Don't use
    # r = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))
    # if x == 0.0:
    #     phi = 0.0
    # else:
    #     phi = np.arctan2(y, x)
    # if r == 0.0:
    #     theta = 0.0
    # else:
    #     theta = np.arccos(z / r)

    # print("\nx={: .3f} y={: .3f} z={: .3f}".format(x, y, z))
    # print("r={: .3f} theta={: .3f} phi={: .3f} (in degrees)".format(r, np.degrees(theta), np.degrees(phi)))
    # print("r={: .8f} theta={: .8f} phi={: .8f} (in radians)".format(r, theta, phi))
    return SlaterOverlap(n1, l1, m1, zeta1, n2, l2, m2, zeta2, r, theta, phi)


if __name__ == "__main__":
    print("---------1--------   ---------2--------  ------")
    print(" n,  l,  m, zeta  n,  l,  m, zeta, dist")
    # print(" 3,  2,  1,  40,  3,  2,  1, 10, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap( 3,  2,  1,  40.0,  3,  2,  1, 10.0, 1.0, 0.0, 0.0), -4.42287766988261e-4))
    # print(" 4,  2,  1, 112,  4,  3,  1, 12, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap( 4,  2,  1, 112.0,  4,  3,  1, 48.0, 1.0, 0.0, 0.0),  4.03505950326382e-17))
    # print(" 5,  4,  4, 170,  5,  4,  4, 30, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap( 5,  4,  4, 170.0,  5,  4,  4, 30.0, 1.0, 0.0, 0.0),  1.56193164484179e-14))
    # print(" 7,  3,  2, 255,  4,  3,  2, 45, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap( 7,  3,  2, 255.0,  4,  3,  2, 45.0, 1.0, 0.0, 0.0),  -1.7686105069788e-18))
    # print(" 9,  5,  3,  54,  8,  4,  3, 36, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap( 9,  5,  3, 54,  8,  4,  3, 36, 1.0, 0.0, 0.0), -5.465102430228e-8))
    # print("10,  7,  1,  72,  8,  2,  1, 48, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(10,  7,  1,  72.0,  8,  2,  1, 48.0, 1.0, 0.0, 0.0), -1.84189026173198e-10))
    # print("10,  9,  9,  24, 10,  9,  9,  6, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(10,  9,  9,  24.0, 10,  9,  9,  6.0, 1.0, 0.0, 0.0),  6.2312231819e-4))
    # print("13, 12, 12,25.25, 13, 12, 12,24.75, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(13, 12, 12,  25.25, 13, 12, 12, 24.75, 1.0, 0.0, 0.0), 1.35310560392e-4))
    # print("14, 13, 13,  21, 14, 13, 13,  9, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(14, 13, 13,  21.0, 14, 13, 13,  9.0, 1.0, 0.0, 0.0),  4.5355131215652e-3))
    # print("15, 14, 14,  15, 15, 14, 14, 15, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(15, 14, 14,  15.0, 15, 14, 14, 15.0, 1.0, 0.0, 0.0),  3.74722497038009e-2))
    # print("16, 15, 15, 35, 16, 15, 15, 35, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(16, 15, 15,  35.0, 16, 15, 15, 35.0, 1.0, 0.0, 0.0),  1.21686562253236e-6))
    # print("17,  8,  4, 55,  8,  7,  4, 45, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(17,  8,  4,  55.0,  8,  7,  4, 45.0, 1.0, 0.0, 0.0), -1.00640061354258e-6))
    # print("17, 16, 16, 12.5, 17, 16, 16, 37.5, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(17, 16, 16,  12.5, 17, 16, 16, 37.5, 1.0, 0.0, 0.0),  3.06769565185575e-5))
    # print("18, 12, 12,   8, 18, 12, 12, 32, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(18, 12, 12,   8.0, 18, 12, 12, 32.0, 1.0, 0.0, 0.0),  6.63931813696651e-5))
    # print("21, 10,  6,  45,  9,  8,  6, 45, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(21, 10,  6,  45.0,  9,  8,  6, 45.0, 1.0, 0.0, 0.0),  5.38980685350612e-5))
    # print("27,  8,  7,  28,  9,  8,  7, 42, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(27,  8,  7,  28.0,  9,  8,  7, 42.0, 1.0, 0.0, 0.0), -1.744238075196959e-4))
    # print("30, 10,  8,  35, 14, 10,  8, 35, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(30, 10,  8,  35.0, 14, 10,  8, 35.0, 1.0, 0.0, 0.0),  1.35074709592800e-2))
    # print("37,  8,  6,   4, 12, 10,  6, 16, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(37,  8,  6,   4.0, 12, 10,  6, 16.0, 1.0, 0.0, 0.0),  3.98219849004259e-14))
    # print("40,  4,  3,  24, 12,  4,  3,  6, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(40,  4,  3,  24.0, 12,  4,  3,  6.0, 1.0, 0.0, 0.0),  9.48379265599810e-2))
    # print("43, 10,  6,  36, 18,  8,  6, 84, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(43, 10,  6,  36.0, 18,  8,  6, 84.0, 1.0, 0.0, 0.0), -1.15907687123104e-4))
    # print("50,  4,  4,42.5, 50,  4,  4,7.5, 1.00: {: .15e} should: {: .15e}.".format(SlaterOverlap(50,  4,  4,  42.5, 50,  4,  4,  7.5, 1.0, 0.0, 0.0),  1.84395901037228e-12))

    print("C 2s overlap with each hydrogen 1s")
    print(" 2,  0,  0,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 0, 0, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, 0.00000, 0.000000, 1.100000),
        0.51331891))
    print(" 2,  0,  0,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 0, 0, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, 1.03709, 0.000000, -0.366667),
        0.51331891))
    print(" 2,  0,  0,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 0, 0, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, -0.51855, 0.898146, -0.366667),
        0.51331891))
    print(" 2,  0,  0,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 0, 0, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, -0.51855, -0.898146, -0.366667),
        0.51331891))

    print("C 2pz overlap with each hydrogen 1s")
    print(" 2,  1,  0,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 1, 0, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, 0.00000, 0.000000, 1.100000),
        0.48549314))
    print(" 2,  1,  0,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 1, 0, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, 1.03709, 0.000000, -0.366667),
        -0.1618))
    print(" 2,  1,  0,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 1, 0, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, -0.51855, 0.898146, -0.366667),
        -0.1618))
    print(" 2,  1,  0,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 1, 0, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, -0.51855, -0.898146, -0.366667),
        -0.1618))

    print("C 2px overlap with each hydrogen 1s")
    print(" 2,  1,  1,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 1, 1, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, 0.00000, 0.000000, 1.100000),
        0.0000))
    print(" 2,  1,  1,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 1, 1, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, 1.03709, 0.000000, -0.366667),
        0.457727343959))
    print(" 2,  1,  1,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 1, 1, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, -0.51855, 0.898146, -0.366667),
        -0.228865383492))
    print(" 2,  1,  1,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 1, 1, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, -0.51855, -0.898146, -0.366667),
        -0.228665383492))

    print("C 2py overlap with each hydrogen 1s")
    print(" 2,  1,  -1,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 1, -1, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, 0.00000, 0.000000, 1.100000),
        0.0000))
    print(" 2,  1,  -1,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 1, -1, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, 1.03709, 0.000000, -0.366667),
        0.0000))
    print(" 2,  1,  -1,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 1, -1, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, -0.51855, 0.898146, -0.366667),
        0.3964))
    print(" 2,  1,  -1,  1.625,  1,  0,   0,  1.200, 2.0787: {: .8f} should: {: .8f}.".format(
        SlaterOverlapCartesian(2, 1, -1, 1.625, 0.00000, 0.00000, 0.00000, 1, 0, 0, 1.2, -0.51855, -0.898146,
                               -0.366667),
        -0.3964))

    print("Demonstration of orthonormality")
    print(" 2,  0,  0,  1.625,  2,  1,   0,  1.625, 0.0000: {: .8f} should: {: .8f}.".format(
        SlaterOverlap(2, 0, 0, 1.625, 2, 1, 0, 1.625, 0.0, 0.0, 0.0), 0.0))
    print(" 2,  1,  0,  1.625,  2,  1,   1,  1.625, 0.0000: {: .8f} should: {: .8f}.".format(
        SlaterOverlap(2, 1, 0, 1.625, 2, 1, 1, 1.625, 0.0, 0.0, 0.0), 0.0))
    print(" 2,  1,  0,  1.625,  2,  1,   0,  1.625, 0.0000: {: .8f} should: {: .8f}.".format(
        SlaterOverlap(2, 1, 0, 1.625, 2, 1, 0, 1.625, 0.0, 0.0, 0.0), 1.0))
    print(" 2,  1,  0,  1.625,  2,  1,  -1,  1.625, 0.0000: {: .8f} should: {: .8f}.".format(
        SlaterOverlap(2, 1, 0, 1.625, 2, 1, -1, 1.625, 0.0, 0.0, 0.0), 0.0))
