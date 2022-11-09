import re
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="post process matchs.")
parser.add_argument("-f", "--file", dest="file", type=str,
                    default="", help="file anme.")
parser.add_argument("-a", "--agent_name", dest="agent", type=str,
                    default="Bob", help="file anme.")
params = parser.parse_args()


def process(file_name, agent_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    patt = re.compile("STATE.+:([-\d]+)\|([-\d]+):(\w+)\|(\w+)")
    scores = np.empty(shape=(len(lines),), dtype=np.int32)
    pos = 0
    for line in lines:
        # print(line)
        result = patt.search(line)
        if result:
            if (result.group(3) == agent_name):
                scores[pos] = int(result.group(1))
            else:
                scores[pos] = int(result.group(2))
            # print(scores[pos])
            pos += 1
    scores = scores[:pos]
    ret = scores.mean(), scores.var(), scores.std()
    with open(file_name, "a") as f:
        f.write(str(ret))
    return ret


if __name__ == "__main__":
    res = process(params.file, params.agent)
    print(res)
