import sys

amount = 5
coins = [1, 2, 3, 4, 5, 6]
ret = 0


def dfs(s):
    global ret
    if sum(s) == amount:
        print(s)
        ret += 1
        return 0
    for c in coins:
        if len(s) < 1 or (c >= s[-1] and sum(s) + c <= amount):
            next_s = s + [c]
            dfs(next_s)


dfs([])
print(ret)
