def solution(A):
    m = sorted(set(A))
    for i in range(len(m) + 1):
        if i + 1 != m[i]:
            return i + 1

res = solution([1, 3, 6, 4, 2])
print(res)
