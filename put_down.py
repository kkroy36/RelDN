#if 3 blocks are stacked then put it down
from random import random

class State(object):
    '''generates states with 1 block or 2 blocks'''

    noise_prob = 0.2

    def __init__(self,number):
        '''constructor'''
        self.action = "putdown(s"+str(number)+")"
        if random() < 0.5:
            self.state = ["on(s"+str(number)+",b2,b1)",
                          "clear(s"+str(number)+",b2)"]
            self.threeStack = True
        else:
            self.state = ["on_table(s"+str(number)+",b1,table)",
                          "clear(s"+str(number)+",b1)"]
            self.threeStack = False

def generateStates():
    '''generates block stacks'''
    facts,pos,neg,examples = [],[],[],[]
    for i in range(30):
        s = State(i)
        for fact in s.state:
            facts.append(fact)
        if not s.threeStack:
            examples.append(s.action+" 1")
            if random() < State.noise_prob:
                neg.append(s.action)
            else:
                pos.append(s.action)
        else:
            examples.append(s.action+" 0")
            if random() < State.noise_prob:
                pos.append(s.action)
            else:
                neg.append(s.action)

    with open("facts.txt","a") as fp:
        for fact in facts:
            fp.write(fact+"\n")
    with open("pos.txt","a") as fp:
        for ex in pos:
            fp.write(ex+"\n")
    with open("neg.txt","a") as fp:
        for ex in neg:
            fp.write(ex+"\n")
    with open("examples.txt","a") as fp:
        for ex in examples:
            fp.write(ex+"\n")

generateStates()
