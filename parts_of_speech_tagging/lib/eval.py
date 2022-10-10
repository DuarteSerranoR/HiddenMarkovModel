

"""
import numpy as np

def unpunct(txt: str):
    return txt.lower() \
             .replace(",","") \
             .replace(";","") \
             .replace(":","") \
             .replace(".","") \
             .replace("?","") \
             .replace("!","") \
             .replace("-","") \
             .replace("—","") \
             .replace("(","") \
             .replace(")","") \
             .replace("\"","") \
             .replace("{","") \
             .replace("}","") \
             .replace("[","") \
             .replace("]","") \
             .replace("  "," ") \
             .replace("  "," ") \
             .strip()


def lev_equal(s1: str,s2: str, tresh: int):
    if levenshtein_distance_dp(s1,s2) <= tresh:
        return True
    else:
        return False


def levenshtein_distance_dp(t1: list, t2: list):
    #t1=t1.split(" ")
    #t2=t2.split(" ")
    distances = np.zeros((len(t1) + 1, len(t2) + 1))

    for i in range(len(t1) + 1):
        distances[i][0] = i

    for i in range(len(t2) + 1):
        distances[0][i] = i

    a = 0
    b = 0
    c = 0

    for i1 in range(1, len(t1) + 1):
        for i2 in range(1, len(t2) + 1):
            if (t1[i1-1] == t2[i2-1]):
                distances[i1][i2] = distances[i1 - 1][i2 - 1]
            else:
                a = distances[i1][i2 - 1]
                b = distances[i1 - 1][i2]
                c = distances[i1 - 1][i2 - 1]
                
                if (a <= b and a <= c):
                    distances[i1][i2] = a + 1
                elif (b <= a and b <= c):
                    distances[i1][i2] = b + 1
                else:
                    distances[i1][i2] = c + 1

    return distances[len(t1)][len(t2)]



def WER(test_txt: str, original_txt: str, n_distance_for_replacements: int = 6, w_punct: bool = False, levenshtein_distance_tresh: int = 3):

    #S=0 # substitutions
    #D=0 # deletions
    #I=0 # insertions
    #C=0 # correct words
    #N=0 # number of words in the reference (N=S+D+C)

    if not w_punct:
        test_txt = unpunct(test_txt)
        original_txt = unpunct(original_txt)

    test_txt="".join(char for char in test_txt if char.isalnum() or char==" ")
    test_words=[ word for word in " ".join(test_txt.splitlines()).split() if word!="" and word!=" " ]
    
    original_txt="".join(char for char in original_txt if char.isalnum() or char==" ")
    original_words=[ word for word in " ".join(original_txt.splitlines()).split() if word!="" and word!=" " ]
    
    #length=max([len(test_words),len(original_words)])
    N=max([len(test_words),len(original_words)])

    e=levenshtein_distance_dp(test_words,original_words) # number of word errors, calculated through distances


    #WER=(S+D+I)/N
    WER=e/N
    return WER

"""






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