from __future__ import print_function

from Utils import Utils
from Tree import node
from Boosting import Boosting
from sys import argv
from os import system

class GradientBoosting(object):

    def __init__(self,regression=False,trees=10,treeDepth=2,loss="LS",sampling_rate=1.0):
        self.targets = None #accept targets for learning
        self.regression = regression #if regression or classification
        self.sampling_rate = sampling_rate #sampling rate of how much of training data to sample
        self.numberOfTrees = trees #number of trees to learn
        self.treeDepth = treeDepth #tree depth
        self.trees = {} #stores all the trees per target
        self.data = None #stores the data used for training
        self.loss = loss #stores the loss function during regression
        self.testPos,self.testNeg,self.testExamples = {},{},{} #stores the positive negative,reg ex's

    def setTargets(self,targets):
        self.targets = targets #sets the targets

    def learn(self,facts=None,examples=None,bk=None,pos=None,neg=None):
	'''learns the regression tree per target'''
        for target in self.targets: #for every target learn
            data = Utils.setTrainingData(target=target,facts=facts,examples=examples,bk=bk,regression=self.regression,sampling_rate = self.sampling_rate,pos=pos,neg=neg)
            trees = [] #initialize place holder for trees
            for i in range(self.numberOfTrees):
                print ('='*20,"learning tree",str(i),'='*20)
                node.setMaxDepth(self.treeDepth) #set max depth of individual tree learned
                node.learnTree(data) #learn the regression tree
                trees.append(node.learnedDecisionTree) 
                Boosting.updateGradients(data,trees,loss=self.loss)
        self.trees[target] = trees
        for tree in trees: #print each tree learned
            print ('='*30,"tree",str(trees.index(tree)),'='*30)
            for clause in tree:
                print (clause)

    def infer(self,facts,examples):
	'''perform inference on learned trees'''
        self.testExamples = {}
        for target in self.targets:
            data = Utils.setTestData(target=target,facts=facts,examples=examples,regression=self.regression)
            Boosting.performInference(data,self.trees[target])
            self.testExamples[target] = data.examples
            print (data.examples)
