import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import rc
mpl.use('Agg')
SPINE_COLOR = 'gray'


def proof_lemma_4():
    n = 10
    r = 100
    lefts = []
    right1s = []
    right2s = []
    right_1_s = []
    right_2_s = []
    for i in range(r):
        a = (np.random.random(n) - 0.5) + (i + 1)
        a_norm_2 = np.linalg.norm(np.maximum(a, 0), ord=2) ** 2
        b = np.random.random(n)
        b_norm_2 = np.linalg.norm(b, ord=2) ** 2
        nL = np.random.random()
        b = b / np.sqrt(b_norm_2) * np.sqrt(a_norm_2 + nL)
        a = np.maximum(a, 0)
        left = np.abs(np.linalg.norm(a, ord=2) ** 2 / np.linalg.norm(a, ord=1) -
                      np.linalg.norm(b, ord=2) ** 2 / np.linalg.norm(b, ord=1))
        right_0 = np.abs(np.linalg.norm(b, ord=1) - np.linalg.norm(a, ord=1))
        right_1 = np.abs(np.linalg.norm(a, ord=2) ** 2 -
                         np.linalg.norm(b, ord=2) ** 2)
        right_2 = right_0 + right_1 / np.linalg.norm(b, ord=1)
        right_3 = right_0 + right_1 / np.linalg.norm(a, ord=1)
        lefts.append(left)
        right1s.append(right_2)
        right2s.append(right_3)
        right_1_s.append(right_0)
        right_2_s.append(right_1 / np.linalg.norm(b, ord=1))
        assert left <= right_2
        assert left <= right_3
    plt.plot(lefts, label="left")
    plt.plot(right1s, label="right 1")
    plt.plot(right2s, label="right 2")
    plt.plot(right_1_s, label="right l 1")
    plt.plot(right_2_s, label="right r 2")
    # plt.ylim([0, 1])
    plt.legend()
    plt.savefig("proof_lemma_4.png")


def proof_prop_1():
    n = 10
    r = 100
    lefts = []
    left1s = []
    right1s = []
    for i in range(r):
        a = np.random.random(n) - 0.5
        r = np.random.random(n) - 0.5
        b = a + r
        # b = np.abs(a + np.vstack(a) * 0.5 + np.random.random(n) * 0.1)
        a = np.maximum(a, 0)
        b = np.maximum(b, 0)
        left1 = np.abs(np.linalg.norm(b, ord=1) - np.linalg.norm(a, ord=1))
        left = np.abs(np.linalg.norm(b - a, ord=1))
        right = np.linalg.norm(r, ord=1)
        lefts.append(left)
        left1s.append(left1)
        right1s.append(right)
        assert left <= right
    plt.plot(left1s, label="left1")
    plt.plot(lefts, label="left")
    plt.plot(right1s, label="right 1")
    plt.legend()
    plt.savefig("proof_prop_1.png")


def proof_lemma_5():
    n = 10
    r = 100
    lefts = []
    left1s = []
    right1s = []
    for i in range(r):
        a = (np.random.random(n) - 0.5) * i
        a_norm_2 = np.linalg.norm(np.maximum(a, 0), ord=2) ** 2
        b = np.random.random(n)
        b_norm_2 = np.linalg.norm(b, ord=2) ** 2
        nL = np.random.random()
        b = b / np.sqrt(b_norm_2) * np.sqrt(a_norm_2 + nL)
        b_norm_2 = np.linalg.norm(b, ord=2) ** 2
        left1 = np.abs(np.linalg.norm(b, ord=1) -
                       np.linalg.norm(np.maximum(a, 0), ord=1))
        left = np.abs(np.linalg.norm(b - a, ord=1))
        lefts.append(left)
        left1s.append(left1)
        right1s.append(b_norm_2 - a_norm_2)
    plt.plot(left1s, label="left1")
    # plt.plot(lefts, label="left")
    plt.plot(right1s, label="right 1")
    plt.legend()
    plt.savefig("proof_lemma_5.png")


if __name__ == "__main__":
    proof_lemma_4()
    # proof_prop_1()
    # proof_lemma_5()
