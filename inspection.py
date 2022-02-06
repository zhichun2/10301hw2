import sys
import math
import numpy as np

# data[-1] to get the last col
# numpy function np.unique to findout majority
# https://numpy.org/doc/stable/reference/generated/numpy.unique.html

# Get all of the arguments from command line and make sure 
# the number of arguments is correct.
# assert(len(sys.argv) == 3)

# setting up global constant
train_file = sys.argv[1]
inspect_file = sys.argv[2]
with open(train_file, 'r') as f_in:
    train = f_in.readlines()

# organize the train data into a numpy array
def organize(data):
    organized = []
    for i in range(1, len(data)):
        organized.append(data[i].rstrip('\n').split('\t'))
    return np.array(organized)

# probability of data[i] == x
def probability(data, i, x):
    d = len(data[data[:, i]== x])
    n = len(data)
    return d/n

def entropy(data):
    print(data.size)
    # if the dataset is empty, return 0 for entropy
    if data.size == 0:
        return 0
    else: 
        print("data right before calling probability")
        print(data)
        p = probability(data, -1, data[0][-1])
        print('p= ' + str(p))
        if p == 1:
            h = -1*(p*math.log2(p))
        elif p == 0:
            h = -1*((1-p)*math.log2(1-p))
        else: 
            h = -1*(p*math.log2(p) + (1-p)*math.log2(1-p))
        return h

# return the label string with majority vote 
def error(data):
    total = len(data)
    values, counts = np.unique(data[:, -1], return_counts=True)
    majority = max(counts[0], counts[1])
    return majority/total

if __name__ == '__main__':
    data = organize(train)
    entropy = entropy(data)
    error = error(data)
    with open(inspect_file, 'w') as f_out:
        f_out.write('entropy: ' + str(entropy) + '\n')
        f_out.write('error: ' + str(error) + '\n')
