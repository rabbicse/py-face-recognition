import math


def solution(A, B):
    result = ''

    ratioA = int(math.ceil(A / B))
    ratioB = int(math.ceil(B / A))

    rA, rB = 0, 0
    while rA < A or rB < B:
        if ratioA > ratioB:
            mA = min(ratioA, A - rA, 2)
            if mA > 0:
                result += 'a' * mA
                rA += mA

            mB = min(ratioB, B - rB, 2)
            if mB > 0:
                result += 'b' * mB
                rB += mB
        else:
            mB = min(ratioB, B - rB, 2)
            if mB > 0:
                result += 'b' * mB
                rB += mB

            mA = min(ratioA, A - rA, 2)
            if mA > 0:
                result += 'a' * mA
                rA += mA
    return result


A = 1
B = 4
result = solution(A, B)
print(result)
