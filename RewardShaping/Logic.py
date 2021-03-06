from __future__ import print_function

import itertools,re
from copy import deepcopy

#Thanks to Chris Meyers for some of this code --> http://www.openbookproject.net/py4fun/prolog/prolog1.html.

class Term(object):
    '''class for term in prolog proof'''
    def __init__ (self, s) :   # expect "x(y,z...)"
        if s[-1] != ')' : fatal("Syntax error in term: %s" % [s])
        flds = s.split('(')
        if len(flds) != 2 : fatal("Syntax error in term: %s" % [s])
        self.args = flds[1][:-1].split(',')
        self.pred = flds[0]

    def __repr__ (self) :
        return "%s(%s)" % (self.pred,",".join(self.args))

class Rule(object):
    '''class for logic rules in prolog proof'''
    def __init__ (self, s) :   # expect "term-:term;term;..."
        flds = s.split(":-")
        self.head = Term(flds[0])
        self.goals = []
        if len(flds) == 2 :
            flds = re.sub("\),",");",flds[1]).split(";")
            for fld in flds : self.goals.append(Term(fld))

    def __repr__ (self) :
        rep = str(self.head)
        sep = " :- "
        for goal in self.goals :
            rep += sep + str(goal)
            sep = ","
        return rep

class Goal(object):
    '''class for each goal in rule during prolog search'''
    def __init__ (self, rule, parent=None, env={}) :
        goalId = Prover.goalId
        goalId += 1
        self.id = goalId
        self.rule = rule
        self.parent = parent
        self.env = deepcopy(env)
        self.inx = 0      # start search with 1st subgoal

    def __repr__ (self) :
        return "Goal %d rule=%s inx=%d env=%s" % (self.id,self.rule,self.inx,self.env)

class Prover(object):
    '''class for prolog style proof of query'''
    rules = []
    goalId = 100
    trace = 0

    @staticmethod
    def unify (srcTerm, srcEnv, destTerm, destEnv) :
        "unification method"
        nargs = len(srcTerm.args)
        if nargs        != len(destTerm.args) : return 0
        if srcTerm.pred != destTerm.pred      : return 0
        for i in range(nargs) :
            srcArg  = srcTerm.args[i]
            destArg = destTerm.args[i]
            if srcArg <= 'Z' :
                srcVal = srcEnv.get(srcArg)
            else             :
                srcVal = srcArg
            if srcVal :    # constant or defined Variable in source
                if destArg <= 'Z' :  # Variable in destination
                    destVal = destEnv.get(destArg)
                    if not destVal :
                        destEnv[destArg] = srcVal  # Unify !
                    elif destVal != srcVal : return 0           # Won't unify
                elif     destArg != srcVal : return 0           # Won't unify
            
        return 1

    @staticmethod
    def search (term) :
        '''method to perform prolog style query search'''
        goalId = Prover.goalId
        trace = Prover.trace
        rules = Prover.rules
        unify = Prover.unify
        goalId = 0
        returnValue = False
        if trace : print("search", term)
        goal = Goal(Rule("got(goal):-x(y)"))      # Anything- just get a rule object
        goal.rule.goals = [term]                  # target is the single goal
        if trace : print("stack", goal)
        stack = [goal]                            # Start our search
        while stack :
            c = stack.pop()        # Next goal to consider
            if trace : print("  pop", c)
            if c.inx >= len(c.rule.goals) :       # Is this one finished?
                if c.parent == None :             # Yes. Our original goal?
                    if c.env : print(c.env)       # Yes. tell user we
                    else     :
                        returnValue = True #print "Yes"        # have a solution
                    continue
                parent = deepcopy(c.parent)  # Otherwise resume parent goal
                unify (c.rule.head,c.env,parent.rule.goals[parent.inx],parent.env)
                parent.inx = parent.inx+1         # advance to next goal in body
                if trace : print("stack", parent)
                stack.append(parent)              # let it wait its turn
                continue
            # No. more to do with this goal.
            term = c.rule.goals[c.inx]            # What we want to solve
            for rule in rules :             # Walk down the rule database
                if rule.head.pred      != term.pred      : continue
                if len(rule.head.args) != len(term.args) : continue
                child = Goal(rule, c) # A possible subgoal
                ans = unify (term, c.env, rule.head, child.env)
                if ans :                            # if unifies, stack it up
                    if trace : print("stack", child)
                    stack.append(child)
        return returnValue

    @staticmethod
    def prove(facts,example,clause):
        '''proves if example satisfies clause given the data
           returns True if satisfies else returns False
        '''
        Prover.rules = [] #contains all rules
        Prover.trace = 0  #if trace is 1 displays proof tree
        Prover.goalId = 100 #stores goal Id
        Prover.rules += [Rule(fact) for fact in facts]
        Prover.rules += [Rule(clause)]
        proofOutcome = Prover.search(Term(example)) #proves query prolog style
        return proofOutcome
