
def WER(txt1: str, txt2: str):
    return levenshtein_distance(txt1.split(" "),txt2.split(" "))

#def FScores():

def levenshtein_distance(x, y):
    n = len(x)
    m = len(y)

    A = [[i + j for j in range(m + 1)] for i in range(n + 1)]

    for i in range(n):
        for j in range(m):
            A[i + 1][j + 1] = min(A[i][j + 1] + 1,              # insert
                                  A[i + 1][j] + 1,              # delete
                                  A[i][j] + int(x[i] != y[j]))  # replace

    return A[n][m]
