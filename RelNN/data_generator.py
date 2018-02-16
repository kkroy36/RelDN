import random

def sample(pdf):
    cdf = [(i, sum(p for j,p in pdf if j < i)) for i,_ in pdf]
    R = max(i for r in [random.random()] for i,c in cdf if c <= r)
    return R

def samp(pots,names):
    probabilities = [x/float(sum(pots)) for x in pots]
    pdf = zip(names,probabilities)
    return sample(pdf)

people = ["p"+str(i+1) for i in range(random.randint(10,20))]
action_movies = ["am"+str(i+1) for i in range(random.randint(10,20))]
drama_movies = ["dm"+str(i+1) for i in range(random.randint(10,20))]

facts = []
examples = []

for m in action_movies:
    facts.append("action("+m+")")
for m in drama_movies:
    facts.append("drama("+m+")")
    
for p in people:
    gender = samp([1,1],["male","female"])
    if gender == "male":
        examples.append("male("+p+") 1")
    else:
        examples.append("male("+p+") 0")
    age = samp([10,5],["young","old"])
    if age == "old":
        facts.append("old("+p+")")
    profession = samp([10,8],["doctor","teacher"])
    if profession == "doctor":
        facts.append("doctor("+p+")")
    else:
        facts.append("teacher("+p+")")
    movies = zip(action_movies,drama_movies)
    for item in movies:
        pref = samp([10,7],["drama","action"])
        if pref == "drama":
            facts.append("likes("+p+","+item[1]+")")
        else:
            facts.append("likes("+p+","+item[0]+")")

with open("test_facts.txt","a") as f:
    for fact in facts:
        f.write(fact+"\n")
with open("test_examples.txt","a") as f:
    for example in examples:
        f.write(example+"\n")
