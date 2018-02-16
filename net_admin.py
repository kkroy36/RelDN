import random
class Node():

    def __init__(self,number):
        self.node_number = number
        self.buffer_capacity = 10 #packets
        self.buffer = 0 #0 packets initially
        self.alive = True

    def buffer_size(self):
        return self.buffer

    def active(self):
        return self.alive

    def add_packet(self):
        self.buffer += 1 #increase buffer size by 1
        if self.buffer > self.buffer_capacity:
            self.alive = False

    def id(self):
        return self.node_number

class Network():

    def __init__(self,number):
        self.network_number = number
        self.nodes = [Node(self.network_number+str(i+1)) for i in range(random.randint(1,5))]
        self.all_connected = (random.random() < 0.7)

    def is_all_connected(self):
        if len(self.get_nodes()) == 1:
            return False
        return self.all_connected

    def get_nodes(self):
        return self.nodes

    def id(self):
        return self.network_number

    def contains(self,node_to_check):
        nodes = self.get_nodes()
        for node in nodes:
            if node.id() == node_to_check.id():
                return True
        return False

    def link_broken(self):
        nodes = self.get_nodes()
        for node in nodes:
            if not node.active():
                return True
        return False

class Admin():

    bk = ["nodeIn(+state,+node,+network)",
          "overloaded(+state,+node)",
          "cyclic(+state,+network)",
          "value(state)"]

    def __init__(self,number=1,start=False):
        if start:
            self.state_number = number
            self.networks = [Network(str(i+1)) for i in range(random.randint(2,4))]
            self.all_actions = []

    def get_all_actions(self):
        self.all_actions = []
        for network in self.networks:
            for node in network.get_nodes():
                self.all_actions.append((network,node))

    def goal(self):
        for network in self.networks:
            if network.link_broken() and not network.is_all_connected():
                return True
        return False

    def execute_action(self,action):
        self.state_number += 1
        network = action[0]
        node = action[1]
        if network.contains(node):
            node.add_packet()
        return self

    def get_state_facts(self):
        facts = []
        for network in self.networks:
            net_id = "net"+network.id()
            if not network.is_all_connected():
                facts.append("cyclic(s"+str(self.state_number)+","+net_id+")")
            for node in network.get_nodes():
                node_id = "node"+node.id()
                facts.append("nodeIn(s"+str(self.state_number)+","+node_id+","+net_id+")")
                #network_facts += [[node.id(),node.active(),node.buffer_size()]]
                if not node.active():
                    facts.append("overloaded(s"+str(self.state_number)+","+node_id+")")
        return facts

    def sample(self,pdf):
        cdf = [(i, sum(p for j,p in pdf if j < i)) for i,_ in pdf]
        R = max(i for r in [random.random()] for i,c in cdf if c <= r)
        return R

    def execute_random_action(self):
        self.get_all_actions()
        N = len(self.all_actions)
        random_actions = []
        action_potentials = []
        for i in range(N):
            random_action = random.choice(self.all_actions)
            random_actions.append(random_action)
            action_potentials.append(random.randint(1,9))
        action_probabilities = [potential/float(sum(action_potentials)) for potential in action_potentials]
        actions_not_executed = [action for action in self.all_actions if action != random_action]
        probability_distribution_function = zip(random_actions,action_probabilities)
        sampled_action = self.sample(probability_distribution_function)
        new_state = self.execute_action(sampled_action)
        return (new_state,[sampled_action],actions_not_executed)
    
with open("net_admin_out.txt","a") as f:
    i = 0
    while i < 1:
        state = Admin(start=True)
        f.write("start state: "+str(state.get_state_facts())+"\n")
        while not state.goal():
            f.write("="*80+"\n")
            state_action_pair = state.execute_random_action()
            state = state_action_pair[0]
            action = state_action_pair[1][0]
            action_node = "node"+str(action[1].id())
            f.write(action_node+"\n")
            f.write(str(state.get_state_facts())+"\n")
        i += 1
