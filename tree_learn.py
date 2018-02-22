from GradientBoosting import GradientBoosting

bk = ["on_table(+state,+block,[table])",
      "clear(+state,+block)",
      "on(+state,-block,+block)",
      "putdown(state)"]

facts,pos,neg = [],[],[]

with open("facts.txt") as f:
    facts = f.read().splitlines()
with open("pos.txt") as f:
    pos = f.read().splitlines()
with open("neg.txt") as f:
    neg = f.read().splitlines()
    
clf = GradientBoosting(treeDepth=2,trees=5,sampling_rate=1.0)
clf.setTargets(["putdown"])
'''
modify the learn function to
accept the java input and remove stuff required
for the python program
1. Full stops.
2. Precomputes.
3. Any initial paramaters before the facts or modes.
'''
clf.learn(facts=facts,pos=pos,neg=neg,bk=bk)

