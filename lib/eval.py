
def WER(txt1: str, txt2: str):
    len1 = len(txt1.split())
    len2 = len(txt2.split())
    #if len1 != len2:
    #    raise Exception("Both input strings need to have the same number of words to get its word error rate!")
    return ( levenshtein_distance(txt1.split(" "),txt2.split(" ")) / max(len1,len2) ) * 100

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


if __name__ == "__main__":
    input = "Hello! How are you?"
    input1 = "Hello! how are you?"
    input2 = "hello how are you you"

    print(WER(input,input))
    print(WER(input,input1))
    print(WER(input,input2))
