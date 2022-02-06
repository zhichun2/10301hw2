import sys
import math
import numpy as np
import inspection

# setting up global constant
assert(len(sys.argv) == 7)
with open(sys.argv[1], 'r') as f_in:
    trainInput= f_in.readlines()
with open(sys.argv[2], 'r') as f_in:
    testInput= f_in.readlines()
maxDepth = int(sys.argv[3])
trainOut = sys.argv[4]
testOut = sys.argv[5]
metricsOut = sys.argv[6]

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self, data, depth):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.leftval = None
        self.rightval = None
        self.depth = depth
        self.data = data




    # calculate mutual information on attribute i in data 
    def mutualInformation(self, i):
        datax0 = self.data[self.data[:, i]==self.data[0][i]]
        datax1 = self.data[self.data[:, i]!=self.data[0][i]] 
        HYX0 = inspection.entropy(datax0)
        HYX1 = inspection.entropy(datax1)
        PX0 = inspection.probability(self.data, i, self.data[0][i])
        PX1 = 1 - PX0
        IYX = inspection.entropy(self.data) - PX0*HYX0 - PX1*HYX1
        return IYX

    # return maximum mutual information and its attibute index in a tuple
    def findmaxMI(self):
        maxMI = -10000000000000
        attr = -1
        for i in range(len(self.data[0])-1): 
            curMI = self.mutualInformation(i)
            if curMI > maxMI:
                maxMI = curMI
                attr = i
        return (maxMI, attr)

    # return the label string with majority vote 
    def majorityVote(self):
        values, counts = np.unique(self.data[:, -1], return_counts=True)
        if values.size == 1:
            return values[0]
        elif counts[0] > counts[1]:
            return values[0]
        # if counts are equal, return alphabetically last
        elif counts[0] == counts[1]:
            return max(values[0], values[1])
        return values[1]

    # checks if the output labels of a dataset is pure
    def labelIsPure(self):
        for i in range(len(self.data)-1):
            if self.data[i][-1] != self.data[i+1][-1]:
                 return False
        return True

    # checks if all values in the attributes are equal
    def identicalAttr(self):
        for i in range(len(self.data[0])-1):
            for j in range(len(self.data[:, i])-1):
                if self.data[:, i][j] != self.data[:, i][j+1]:
                    return False
        return True

    # checks of base case constrains are satisfied
    def stopCiteria(self, maxDepth):
        if self.data == []:
            return True        
        elif self.depth >= maxDepth:
            return True
        elif self.labelIsPure():
            print("***************************label is pure!")
            return True
        elif self.identicalAttr():
            return True
        elif self.findmaxMI()[0] <= 0:
            return True
        return False

# map the index of attrbutes to the string of attributes
def organizeAttr(data):
    d = {}
    attributes = data[0].rstrip('\n').split('\t')
    for i in range(len(attributes)):
        d[i] = attributes[i]
    return d

# build the tree with data and root node
def train(node, maxDepth):
    if node.stopCiteria(maxDepth):
        node.vote = node.majorityVote()
        return node
    else:
        # creating left and right node
        (maxMI, attr) = node.findmaxMI()
        dataLeft = node.data[node.data[:, attr]==node.data[0][attr]]
        dataRight = node.data[node.data[:, attr]!=node.data[0][attr]]
        leftNode = Node(np.array(dataLeft), node.depth+1)
        rightNode = Node(np.array(dataRight), node.depth+1)
        # are np array still np array after splitting?

        # filling in attributes for current node
        node.attr = attr
        node.leftval = dataLeft[0][attr]
        node.rightval = dataRight[0][attr]   
        node.left = leftNode
        node.right = rightNode

        # recurse 
        train(node.left, maxDepth)
        train(node.right, maxDepth)

        return node

# input tree root, a dictionary of attributes of an entry, and a dict mapping 
# attributes index to attributes string
def predict(node, example, attrMap):
    if node.left == None and node.right == None:
        return node.vote
    else:
        attr = attrMap[node.attr] 
        if example[attr] == node.leftval:
            predict(node.left, example, attrMap)
        else:
            predict(node.right, example, attrMap)

# organize test input to an array of dictionaries??
# is it better to organize the train input into dictionaries or 2d array?
def organizeExamples(data, attrMap):
    examples = []
    for i in range(len(data)):
        e = {}
        for j in range(len(data[i])):
            e[attrMap[j]] = data[i][j]
        examples.append(e)
    return examples

if __name__ == '__main__':
    trainAttr = organizeAttr(trainInput)
    trainData = inspection.organize(trainInput)
    trainExample = organizeExamples(trainData, trainAttr)
    
    root = Node(trainData, 0)
    train(root, maxDepth)
    with open(trainOut, 'w') as f_out:
        for example in trainExample:
            prediction = predict(root, example, trainAttr)
            f_out.write(str(prediction) + '\n')

    testAttr = organizeAttr(testInput)
    testData = inspection.organize(testInput)
    testExample = organizeExamples(testData, testAttr)
    with open(testOut, 'w') as f_out:
        for example in testExample:
            prediction = predict(root, example, testAttr)
            f_out.write(str(prediction) + '\n')
