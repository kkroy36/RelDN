from blocks import Blocks_world
from net_admin import Admin
from time import clock
from GradientBoosting import GradientBoosting

class IRL(object):

    def __init__(self,transfer=0,simulator="blocks",batch_size=1,number_of_iterations=10,loss="LS",trees=10):
        self.transfer = transfer
        self.simulator = simulator
        self.batch_size = batch_size
        self.loss = loss
        self.trees = trees
        self.number_of_iterations = number_of_iterations
        self.model = None
        self.state_number = 1

    def compute_value_of_trajectory(self,values,trajectory,discount_factor=0.9,goal_value=1,pi=False): 
        reversed_trajectory = trajectory[::-1]
        number_of_transitions = len(reversed_trajectory)
        if not pi:
            for i in range(number_of_transitions):
                state_number = reversed_trajectory[i][0]
                state = reversed_trajectory[i][1]
                value_of_state = (goal_value)*(discount_factor**i) #immediate reward 0
                key = (state_number,tuple(state))
                values[key] = value_of_state

    def start(self):
        facts,examples,bk = [],[],[]
        i = 0
        values = {}
        while i < self.transfer*5+3:
            print (i)
            if self.simulator == "blocks":
                state = Blocks_world(number=self.state_number,start=True)
                if not bk:
                    bk = Blocks_world.bk
            elif self.simulator == "net_admin":
                state = Admin(number=self.state_number,start=True)
                if not bk:
                    bk = Admin.bk
            with open(self.simulator+"_transfer_out.txt","a") as f:
                if self.transfer:
                    f.write("start state: "+str(state.get_state_facts())+"\n")
                time_elapsed = 0
                within_time = True
                start = clock()
                trajectory = [(state.state_number,state.get_state_facts())]
                while not state.goal():
                    if self.transfer:
                        f.write("="*80+"\n")
                    state_action_pair = state.execute_random_action()
                    state = state_action_pair[0]
                    if self.transfer:
                        f.write(str(state.get_state_facts())+"\n")
                    trajectory.append((state.state_number,state.get_state_facts()))
                    end = clock()
                    time_elapsed = abs(end-start)
                    if self.simulator == "net_admin" and time_elapsed > 1:
                        within_time = False
                        break
                if within_time:
                    self.compute_value_of_trajectory(values,trajectory)
                    self.state_number += len(trajectory)+1
                    for key in values:
                        facts += list(key[1])
                        example_predicate = "value(s"+str(key[0])+") "+str(values[key])
                        examples.append(example_predicate)
                    i += 1

        '''
        with open("facts.txt","a") as f:
            for fact in facts:
                f.write(fact+"\n")
        with open("examples.txt","a") as f:
            for example in examples:
                f.write(example+"\n")
                    
        '''
        reg = GradientBoosting(regression=True,treeDepth=3,trees=self.trees,sampling_rate=0.1,loss=self.loss)
        reg.setTargets(["value"])
        reg.learn(facts,examples,bk)
        self.model = reg    
