import string
import re
from math import exp
from random import sample
import random

class Data(object):
    '''contains the relational data'''

    def __init__(self):
        '''constructor for the Data class'''
        self.regression = False #flag for regression
        self.advice = False #flag for advice
        self.adviceClauses = {} #advice clauses stored here
        self.facts = [] #facts
        self.facts_in_bk = []
        self.pos = {} #positive examples
        self.neg = {} #negative examples
        self.examples = {} #for regression
        self.examplesTrueValue = {} #for regression
        self.target = None #target to be learned
        self.literals = {} #literals present in facts and their type specs
        self.variableType = {} #type of variable used for facts and target

    def setFacts(self,facts):
        '''set facts from facts list'''
        self.facts = facts
        
    def setFactsinbk(self,facts_in_bk):
        '''set whole facts to fetch the range of constants'''
        self.facts_in_bk = facts_in_bk

    def getFacts(self):
        '''returns the facts in the data'''
        return self.facts

    def setPos(self,pos,target):
        '''set positive examples from pos list'''
        for example in pos:
            if example.split('(')[0] == target:
                self.pos[example] = 0.1192 #set initial gradient to 1-0.5 for positive

    def setExamples(self,examples,target):
        '''set examples for regression'''
        for example in examples:
            predicate = example.split(' ')[0] #get predicate
            value = float(example.split(' ')[1]) # get true regression value
            if predicate.split('(')[0] == target:
                self.examplesTrueValue[predicate] = value #store true value of example
                self.examples[predicate] = value #set value for example, otherwise no variance

    def setNeg(self,neg,target):
        '''set negative examples from neg list'''
        for example in neg:
            if example.split('(')[0] == target:
                self.neg[example] = -0.8808 #set initial gradient to 0-0.5 for negative

    def setTarget(self,bk,target,regression = False):
        '''sets the target'''
        targetSpecification = None
        for line in bk:
            if line.split('(')[0] == target:
                targetSpecification = line
        targetSpecification = targetSpecification[:-1].split('(')[1].split(',')
        firstPositiveInstance = None
        if not regression:
            for posEx in self.pos.keys(): #get the first positive example in the dictionary
                if posEx.split('(')[0] == target:
                    firstPositiveInstance = posEx
                    break
        elif regression:
            for example in self.examples.keys(): #get first regression example
                predicate = example.split(' ')[0]
                if predicate.split('(')[0] == target:
                    firstPositiveInstance = predicate
                    break
        targetPredicate = firstPositiveInstance.split('(')[0] #get predicate name
        targetArity = len(firstPositiveInstance.split('(')[1].split(',')) #get predicate arity
        targetVariables = sample(Utils.UniqueVariableCollection,targetArity) #get some variables according to arity
        self.target = targetPredicate+"(" #construct target string
        for variable in targetVariables:
            self.target += variable+","
            self.variableType[variable] = targetSpecification[targetVariables.index(variable)]
        self.target = self.target[:-1]+")"

    def getTarget(self):
        '''returns the target'''
        return self.target

    def getExampleTrueValue(self,example):
        '''returns true regression value of example during regression'''
        return self.examplesTrueValue[example]

    def getValue(self,example):
        '''returns regression value for example'''
        if Utils.data.regression:
            return self.examples[example]
        for ex in self.pos: #check first among positive examples and return value
            if ex == example:
                return self.pos[example]
        for ex in self.neg: #check next among negative examples and return values
            if ex == example:
                return self.neg[example]

    def setBackground(self,bk):
        '''obtains the literals and their type specifications
           types can be variable or a list of constants
        '''
        bkWithoutTargets = [line for line in bk if '+' in line or '-' in line]
        for literalBk in bkWithoutTargets: #for every literal obtain name and type specification
            literalName = literalBk.split('(')[0]
            literalTypeSpecification = literalBk[:-1].split('(')[1].split(',')
            self.literals[literalName] = literalTypeSpecification
            
    def getLiterals(self):
        '''gets all the literals in the facts'''
        return self.literals
        
   
        
class Utils(object):
    '''class for utilities used by program
       reading files
    '''

    """
    
    'string' module can cause compatability issues between Python 2 and Python 3,
    switched from using string.uppercase to using string.ascii_uppsercase,
    the latter should work with both versions.
    """

    data = None #attribute to store data (facts,positive and negative examples)
    UniqueVariableCollection = set(list(string.ascii_uppercase))

    @staticmethod
    def addVariableTypes(literal):
        '''adds type of variables contained in literal'''
        literalName = literal.split('(')[0] #get literal name
        literalTypeSpecification = Utils.data.literals[literalName] #get background info
        literalArguments = literal[:-1].split('(')[1].split(',') #get arguments
        numberOfArguments = len(literalArguments)
        for i in range(numberOfArguments):
            if literalTypeSpecification[i][0]!='[':
                variable = literalArguments[i]
                if variable not in Utils.data.variableType.keys():
                    Utils.data.variableType[variable] = literalTypeSpecification[i][1:]

    @staticmethod
    def find_nth(haystack, needle, n):
        
        ''' returns the nth occurence of a character in the string'''
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start+len(needle))
            n -= 1
        return start

    @staticmethod
    def addConstants(constant_position,constant_predicate,facts):
        
        '''returns unique constants for constants in bk file'''
        constant_list = []
        [constant_list.append(each_fact[Utils.find_nth(each_fact,",",constant_position-1)+1:len(each_fact)-1].strip()) for each_fact in facts if each_fact[0:each_fact.index("(")]==constant_predicate[0:constant_predicate.index("(")] and each_fact[Utils.find_nth(each_fact,",",constant_position-1)+1:len(each_fact)-1].strip() not in constant_list]
        
        return "["+";".join(constant_list)+"]"
        
    
        
    @staticmethod
    def getleafValue(examples):
        '''returns average of regression values for examples'''
        if not examples:
            return 0
        total = 0
        for example in examples:
            total += Utils.data.getValue(example)
        return total/float(len(examples))
        
    @staticmethod
    def write_for_java(info_list,filename):
        
        with open (filename,"w") as fp:
            for each in info_list:
                fp.write(each+"."+"\n")
        fp.close()

    @staticmethod
    def setTrainingData(target=None,facts=None,examples=None,pos=None,neg=None,bk=None,regression=False,sampling_rate=None):
        '''sets facts, examples and background'''
        Utils.data = Data()
        Utils.data.regression = regression
        sampled_facts = [fact for fact in facts if random.random() < sampling_rate]
        Utils.data.setFacts(sampled_facts)
        if not regression:
            sampled_pos = [ex for ex in pos if random.random() < sampling_rate]
            sampled_neg = [ex for ex in neg if random.random() < sampling_rate]
            Utils.data.setPos(sampled_pos,target)
            Utils.data.setNeg(sampled_neg,target)
        elif regression:
            sampled_examples = [example for example in examples if random.random() < sampling_rate]
            Utils.data.setExamples(sampled_examples,target)
        Utils.data.setBackground(bk)
        if not regression:
            Utils.data.setTarget(bk,target)
        elif regression:
            Utils.data.setTarget(bk,target,regression = True)
        return Utils.data
    
    @staticmethod
    def readTrainingData(target,sampling_rate_train,regression = False,advice=False):
        '''reads the training data from files'''
        Utils.data = Data() #create object to hold data for each tree
        Utils.data.regression = regression
        Utils.data.advice = advice
        if advice:
            with open("train/advice.txt") as fp: #read advice from train folder
                adviceFileLines = fp.read().splitlines()
                for line in adviceFileLines:
                    adviceClause = line.split(' ')[0] #get advice clause
                    Utils.data.adviceClauses[adviceClause] = {}
                    preferredTargets = line.split(' ')[1][1:-1].split(',')
                    if preferredTargets[0]:
                        Utils.data.adviceClauses[adviceClause]['preferred'] = preferredTargets
                    elif not preferredTargets[0]:
                        Utils.data.adviceClauses[adviceClause]['preferred'] = []
                    nonPreferredTargets = line.split(' ')[2][1:-1].split(',')
                    if nonPreferredTargets[0]:
                        Utils.data.adviceClauses[adviceClause]['nonPreferred'] = nonPreferredTargets
                    elif not nonPreferredTargets[0]:
                        Utils.data.adviceClauses[adviceClause]['nonPreferred'] = []
                    
        with open("train/facts.txt") as fp: #read facts from train folder
            facts = fp.read().replace(".","").replace(" ","").splitlines()
            facts = list (filter (lambda each_fact: not each_fact.startswith("//"),facts))
            #print ("The total number of training facts: ",len(facts))
            len_facts = int(round(len(facts)*(sampling_rate_train)/100))
            sampled_facts = []
            for i in range(len_facts):
                random_index = random.randrange(len(facts))
                sampled_facts.append(facts[random_index])
                del facts[random_index]
            #print ("The facts used in learning---->",sampled_facts)
            print ("The number of sampled training facts: ",len(sampled_facts))
            Utils.write_for_java(sampled_facts,"java_code/train/train_facts.txt")
            Utils.data.setFacts(sampled_facts)
            Utils.data.setFactsinbk(facts)
            for line in sampled_facts:
                print ("fact line: ",line)
            print ("\n")
        #['putdown(state)', 'ontable(+state,+block,[table])', 'on(+state,-block,+block)']
        if not regression:
            with open("train/pos.txt") as fp: #read positive examples from train folder
                pos = fp.read().replace(".","").replace(" ","").splitlines()
                pos = list (filter (lambda each_pos: not each_pos.startswith("//"),pos))
                len_pos = int(round(len(pos)*(sampling_rate_train)/100))
                sampled_pos = []
                #sampled_pos =  [ pos[random.randrange(len(pos))]  for i in range(len_pos)]
                for i in range(len_pos):
                    random_index = random.randrange(len(pos))
                    sampled_pos.append(pos[random_index])
                    del pos[random_index]
                #print (sampled_pos)
                Utils.write_for_java(sampled_pos,"java_code/train/train_pos.txt")
                Utils.data.setPos(sampled_pos,target)
                for line in sampled_pos:
                    print ("pos line: ",line)
                print ("\n")
            with open("train/neg.txt") as fp: #read negative examples from train folder
                neg = fp.read().replace(".","").replace(" ","").splitlines()
                neg = list (filter (lambda each_neg: not each_neg.startswith("//"),neg))
                len_neg = int(round(len(neg)*(sampling_rate_train)/100))
                sampled_neg = []
                for i in range(len_neg):
                    random_index = random.randrange(len(neg))
                    sampled_neg.append(neg[random_index])
                    del neg[random_index]
                #print (sampled_neg)
                Utils.write_for_java(sampled_neg,"java_code/train/train_neg.txt")
                Utils.data.setNeg(sampled_neg,target)
                for line in sampled_neg:
                    print ("neg line: ",line)
                print ("\n")
        elif regression:
            with open("train/examples.txt") as fp: #read training examples for regression
                examples = fp.read().splitlines()
                Utils.data.setExamples(examples,target)
        with open("train/bk.txt") as fp: #read background information from train folder
            bk = fp.readlines()
            precomputes = [s[0:s.index("(")] for s in bk if ":-" in s]
            bk = [s[s.index(":")+1:len(s)].
                    replace(".","").replace("//","").replace(" ","").strip()  for s in bk if s.startswith("mode:")  and "recursive" not in s and s[s.index(":")+1:s.index("(")].replace(".","").replace("//","").strip() not in precomputes]
            #constant = [s for s in bk if "#" in s]
            
            for each_predicate in bk:
                if '#' in each_predicate:
                    bk.remove(each_predicate)
                    constant_position = each_predicate.split("#")[0].count(",")+1
                    #print (each_predicate.split("#")[0])
                    bk.append(each_predicate.split("#")[0] +str(Utils.addConstants(constant_position,each_predicate,Utils.data.facts_in_bk)).replace("\'","").replace(" ","")+")")
                    #exit()
            for line in bk:
                print (line)
            #exit()
            
            Utils.data.setBackground(bk)
            if not regression:
                Utils.data.setTarget(bk,target)
            elif regression:
                Utils.data.setTarget(bk,target,regression = True)
        return Utils.data

    @staticmethod
    def setTestData(target=None,facts=None,pos=None,neg=None,examples=None,regression=False):
        testData = Data()
        testData.regression = regression
        testData.setFacts(facts)
        if not regression:
            testData.setPos(pos)
            testData.setNeg(neg)
        elif regression:
            testData.setExamples(examples,target)
        return testData

    @staticmethod
    def readTestData(target,sampling_rate_test,regression = False):
        '''reads the testing data from files'''
        testData = Data() #create object to hold data
        testData.regression = regression
        with open("test/facts.txt") as fp:
            testData.setFacts(fp.read().replace(".","").replace("//","").splitlines()) #read facts from test folder
        if not regression:
            with open("test/pos.txt") as fp:
                testData.setPos(fp.read().replace(".","").replace("//","").splitlines(),target) #read positive examples from test folder
            with open("test/neg.txt") as fp:
                testData.setNeg(fp.read().replace(".","").replace("//","").splitlines(),target) #read negative examples from test folder
        elif regression:
            with open("test/examples.txt") as fp: #read testing examples for regression
                examples = fp.read().splitlines()
                testData.setExamples(examples,target)
        return testData #return the data for testing

    @staticmethod
    def variance(examples):
        '''method to calculate variance
           in regression values for all
           examples
        '''
        if not examples:
            return 0
        total = 0 #initialize total regression value 
        for example in examples:
            total += Utils.data.getValue(example) #cimpute total
        numberOfExamples = len(examples) #get number of examples
        mean = total/float(numberOfExamples) #calc mean as total/number
        sumOfSquaredError = 0 #initialize sum of squared errors
        for example in examples: #calculate total squared difference from mean
            sumOfSquaredError += (Utils.data.getValue(example)-mean)**2
        return sumOfSquaredError/float(numberOfExamples) #return variance

    @staticmethod
    def sigmoid(x):
        '''returns sigmoid of x'''
        return exp(x)/float(1+exp(x))

    @staticmethod
    def cartesianProduct(itemSets):
        '''returns cartesian product of all the sets
           contained in the item sets
        '''
        modifiedItemSets = [] #have to create new input where each single element is in its own set
        for itemSet in itemSets:
            modifiedItemSet = []
            for element in itemSet:
                modifiedItemSet.append([element]) #does the above task
            modifiedItemSets.append(modifiedItemSet)
        while len(modifiedItemSets) > 1: #perform cartesian product of first 2 sets
            set1 = modifiedItemSets[0]
            set2 = modifiedItemSets[1]
            pairWiseProducts = []
            for item1 in set1:
                for item2 in set2:
                    pairWiseProducts.append(item1+item2) #cartesian product performed here
            modifiedItemSets.remove(set1) #remove first 2 sets
            modifiedItemSets.remove(set2)
            modifiedItemSets.insert(0,pairWiseProducts) #insert cartesian product in its place and repeat
        return modifiedItemSets[0] #return the final cartesian product sets
            
