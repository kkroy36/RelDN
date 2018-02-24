import random
from Logic import Prover

class world(object):

    def __init__(self,n_trucks=2,n_boxes=3,n_cities=3,number=1):
        '''generates an initial state'''
        self.state_number = number
        self.city,self.trucks,self.unloaded_boxes = {},{},{}
        for i in range(n_cities):
            self.city[i+1] = []
            self.unloaded_boxes[i+1] = []
        for i in range(n_trucks):
            self.trucks[i+1] = []
            self.city[1] += [i+1]
        for i in range(n_boxes):
            truck_id = random.randint(1,n_trucks)
            self.trucks[truck_id] += [i+1]

    def move_truck(self,city_id):
        '''moves truck to next city'''
        self.state_number += 1
        if city_id == 3:
            return self
        if not self.city[city_id]:
            return self
        for truck_id in self.city[city_id]:
            self.city[city_id].remove(truck_id)
            self.city[city_id+1].append(truck_id)
            return self
        return self

    def unload_box(self,city_id):
        '''unloads box from truck'''
        self.state_number += 1
        truck_id = None
        if not self.city[city_id]:
            return self
        for city in self.city:
            if self.city[city]:
                truck_id = self.city[city][0]
                break
        box_id = None
        for truck in self.trucks:
            if self.trucks[truck]:
                box_id = self.trucks[truck][0]
                self.trucks[truck].remove(box_id)
        self.unloaded_boxes[city_id].append(box_id)
        return self

    def goal(self):
        '''checks if goal reached'''
        if self.unloaded_boxes[3]:
            return True
        return False

    def get_facts(self):
        '''gets world facts'''
        facts = ["destination(s"+str(self.state_number)+",c3)"]
        for city in self.city:
            for truck in self.city[city]:
                facts.append("tIn(s"+str(self.state_number)+",t"+str(truck)+",c"+str(city)+")")
        for truck in self.trucks:
            for box in self.trucks[truck]:
                facts.append("bOn(s"+str(self.state_number)+",b"+str(box)+",t"+str(truck)+")")
        for city in self.unloaded_boxes:
            for box in self.unloaded_boxes[city]:
                facts.append("bIn(s"+str(self.state_number)+",b"+str(box)+",c"+str(city)+")")
        return facts

def compute_shaping_value(facts,action):
    '''computes shaping reward based on shaping function'''
    f_lines,f_value = [],0
    with open("shaping_function.txt") as f:
        f_lines = f.read().splitlines()
    for f_line in f_lines:
        potential = float(f_line.split(" ")[0])
        clause = f_line.split(" ")[1].replace(";",",")
        action_literal = clause.split(":-")[0].split("(")[0]
        if action_literal == action.split("(")[0]:
            if Prover.prove(facts,action,clause):
                f_value = potential
    return f_value

def compute_tree_distance_value(facts,action):
    '''computes shaping reward based on shaping function'''
    f_lines,f_value = [],0
    with open("tree_distance.txt") as f:
        f_lines = f.read().splitlines()
    for f_line in f_lines:
        potential = float(f_line.split(" ")[0])
        clause = f_line.split(" ")[1].replace(";",",")
        action_literal = clause.split(":-")[0].split("(")[0]
        if action_literal == action.split("(")[0]:
            if Prover.prove(facts,action,clause):
                f_value = potential
    return f_value
        

def create_data_set(tas,shaping=False,tree_distance=False):
    '''create Q-estimation data set'''
    facts,examples = [],[]
    discount_factor = 0.97
    goal_value = 1
    for ta in tas:
        #print ("="*80+"\n")
        #print (ta)
        r_ta = ta[::-1]
        N = len(r_ta)
        for i in range(N):
            facts += r_ta[i][1]
            action = r_ta[i][2]
            q_value = (discount_factor**i)*goal_value
            if shaping:
                f_value = compute_shaping_value(facts,action)
                q_value += f_value
            if tree_distance:
                f_value = compute_tree_distance_value(facts,action)
                q_value += f_value
            examples.append(action+" "+str(q_value))
    with open("facts.txt","a") as f:
        for fact in facts:
            f.write(fact+"\n")
    with open("examples.txt","a") as f:
        for example in examples:
            f.write(example+"\n")
            
s_number = 1
tas = []
for i in range(20):
    s = world(number=s_number)
    if i > 14:
        chance_prob = 0.5
    else:
        chance_prob = 0.8
    t = [s.get_facts()]
    ta = []
    while not s.goal():
        city = random.randint(1,3)
        prob = random.random()
        if city == 1:
            if prob < chance_prob:
                if not s.city[2] and not s.city[3]:
                    #print ("moving in city correctly",1)
                    ta.append((s.state_number,s.get_facts(),"move(s"+str(s.state_number)+")"))
                    s = s.move_truck(1)
                    t.append(s.get_facts())
            else:
                random_choice = random.random()
                if random < 0.5:
                    #print ("unloading in city",1)
                    ta.append((s.state_number,s.get_facts(),"unload(s"+str(s.state_number)+")"))
                    s = s.unload_box(1)
                    t.append(s.get_facts())
                else:
                    #print ("moving in city",1)
                    ta.append((s.state_number,s.get_facts(),"move(s"+str(s.state_number)+")"))
                    s = s.move_truck(1)
                    t.append(s.get_facts())
        elif city == 2:
            if prob < chance_prob:
                if s.city[2] and not s.city[3]:
                    #print ("moving in city correctly",2)
                    ta.append((s.state_number,s.get_facts(),"move(s"+str(s.state_number)+")"))
                    s = s.move_truck(2)
                    t.append(s.get_facts())
            else:
                random_choice = random.random()
                if random < 0.5:
                    #print ("moving in city",2)
                    ta.append((s.state_number,s.get_facts(),"move(s"+str(s.state_number)+")"))
                    s = s.move_truck(2)
                    t.append(s.get_facts())
                else:
                    #print ("unloading in city",2)
                    ta.append((s.state_number,s.get_facts(),"unload(s"+str(s.state_number)+")"))
                    s = s.unload_box(2)
                    t.append(s.get_facts())
        elif city == 3:
            if prob < chance_prob:
                if s.city[3]:
                    #print ("unloading in city correctly",3)
                    ta.append((s.state_number,s.get_facts(),"unload(s"+str(s.state_number)+")"))
                    s = s.unload_box(3)
                    t.append(s.get_facts())
            else:
                random_choice = random.random()
                if random < 0.5:
                    #print ("moving in city",3)
                    ta.append((s.state_number,s.get_facts(),"move(s"+str(s.state_number)+")"))
                    s = s.move_truck(3)
                    t.append(s.get_facts())
                else:
                    #print ("unloading in city",3)
                    ta.append((s.state_number,s.get_facts(),"unload(s"+str(s.state_number)+")"))
                    s = s.unload_box(3)
                    t.append(s.get_facts())
    tas.append(ta)
    s_number += len(t)
create_data_set(tas,shaping=False,tree_distance=True)
