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
clf.learn(facts=facts,pos=pos,neg=neg,bk=bk)

