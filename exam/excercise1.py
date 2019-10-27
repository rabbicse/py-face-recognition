def solution(A, B):
    merged = [(A[i], B[i]) for i in range(len(A))]
    print(merged)

    indices = list(range(len(merged)))
    print(indices)
    # overlap
    tups = []
    processed = []
    for i in indices:
        for j in indices[1:]:
            if i == j:
                continue
            if j in processed or i in processed:
                continue
            tup = (min(merged[i][0], merged[j][0]), max(merged[i][1], merged[j][1]))

            if max(merged[i][0], merged[j][0]) <= min(merged[i][1], merged[j][1]):
                tups.append(tup)

                processed.append(i)
                processed.append(j)

    for m in indices:
        if m not in processed:
            tups.append(merged[m])
    return len(tups)


A = [1, 12, 42, 70, 36, -4, 43, 15]

B = [5, 15, 44, 72, 36, 2, 69, 24]

l = solution(A, B)
print(l)