import matplotlib.pyplot as plt
import pandas as pd


def plot(peg_left_list):
    for number in peg_left_list:
        x,y = number
        plt.plot(x,y)
        plt.show()
        #print(number)