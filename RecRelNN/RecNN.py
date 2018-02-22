from math import exp
from Logic import Prover
from copy import deepcopy
#make modifications in the learn function beyond forward prop

class Node(object):

    def __init__(self,label,clause,ip=False,op=False):
        '''constructor for node in a network'''
        self.label = label
        self.clause = clause
        self.ip = ip
        self.op = op
        self.detla = None
        if ip:
            self.value = None
        self.connections = []

    def set_delta(self,value):
        '''set gradient value'''
        self.detla = value

    def get_delta(self):
        '''get gradient value'''
        return self.delta

    def add_connection(self,node,weight):
        '''add a new node as a connection'''
        self.connections.append([node,weight])

    def get_connection_weight(self,node):
        '''for all connections
           get connected node and
           get connection weight if match
        '''
        for connection in self.get_connections():
            connected_node = connection[0]
            if node.get_label() == connected_node.get_label():
                weight = connection[1]
                return weight
            
    def get_connections(self):
        '''return all connected nodes'''
        return self.connections

    def is_connection(self,node):
        '''check if current node connected to node in param'''
        for connection in self.get_connections():
            connected_node = connection[0]
            if node.get_label() == connected_node.get_label():
                return True
        return False

    def get_value(self):
        '''get output at node'''
        return self.value

    def set_value(self,value):
        '''set output at node'''
        self.value = value

    def get_label(self):
        '''get identifier of node'''
        return self.label

    def get_clause(self):
        '''get clause at node (might be horn with no body)'''
        return self.clause

    def is_input(self):
        '''return if node is in input layer'''
        return self.ip

    def is_output(self):
        '''return if node in output layer'''
        return self.op

    def __repr__(self):
        return self.get_label()+"(\""+self.get_clause()+"\")"

class RecRelNN(object):

    def __init__(self,description_file,learning_rate=0.01,horizon_length=3):
        '''constructor for Recurrent RelNN'''
        self.nodes = []
        self.horizon_length = horizon_length
        self.read_network_description(description_file)
        self.train_facts = []
        self.train_examples = []
        self.learning_rate = learning_rate
        self.test_facts = []
        self.test_examples = []

    def activation(self,x):
	'''activation function at nodes'''
        return exp(x)/float(1+exp(x)) #sigmoid

    def common_connection(self,s,node2):
        node2_connections = node2.get_connections()
        node2_connections = [str(n[0]) for n in node2_connections]
        for node in s:
            node_connections = node.get_connections()
            c = len([n for n in node_connections if str(n[0]) in node2_connections])
            if not c:
                return False
        return True

    def finished_computation_on_all_nodes(self):
	'''checks if all node values have been computed'''
        for node in self.get_nodes():
            if node.get_value() == None:
                return False
        return True

    def get_sets_ip_nodes(self):
        nodes = self.get_nodes()
        sets = []
        for node in nodes:
            if node.is_input():
                sets.append([node])
        for node in nodes:
            if not node.is_input():
                continue
            for s in sets:
                if self.common_connection(s,node):
                    if node not in s:
                        s.append(node)
        sets = [sorted(x) for x in sets]
        u_sets = []
        for s in sets:
            if s not in u_sets:
                u_sets.append(s)
        return u_sets

    def compute_ip_node_values(self,example,test=False):
        '''computes value of input nodes using proof tree'''
        predicate = example.split(" ")[0]
        facts = self.train_facts
        if test:
            facts = self.test_facts
        literal_name = predicate.split("(")[0]
        predicate_time_label = predicate[-2]
        clause_head = ""
        nodes = self.get_nodes()
        for node in nodes:
            rule = node.get_clause()
            rule_body = rule.split(";")
            for item in rule_body:
                item_name = item.split("(")[0]
                item_time_label = item[-2]
                if item_name == literal_name and item_time_label == predicate_time_label:
                    clause_head = item
        sets = self.get_sets_ip_nodes()
        for s in sets:
            clause_body = ""
            for node in s:
                if node.is_input():
                    clause_body += node.get_clause().replace(";",",")+","
            clause = clause_head+":-"+clause_body[:-1]
            if clause_body[:-1] != "true":
                truth = Prover.prove(facts,predicate,clause)
            for node in s:
                if node.is_input():
                    node.set_value(int(truth))

    def learn(self,facts,examples):
        '''learns weights through backprop'''
        self.set_train_facts(facts)
        self.set_train_examples(examples)
        for i in range(20): #until backprop epsilon convergence
            for example in self.train_examples:
                print (example)
                self.compute_ip_node_values(example)
                self.forward_propogate()
        nodes = self.get_nodes()
        for node in nodes:
	    print (node,node.get_value())
	exit()

    def get_incoming(self,node):
	'''gets all incoming connection nodes'''
        incoming_nodes = []
        for n in self.get_nodes():
            if n.is_connection(node):
                incoming_nodes.append(n)
        return incoming_nodes

    def compute_node_value(self,node,incoming_nodes):
	'''computes value as weighted sum of incoming connections'''
        total = 0
        if node.is_input():
            return
        for n in incoming_nodes:
            if n.get_value() == None:
                return
            elif n.get_value():
                value = n.get_value()
                weight = n.get_connection_weight(node)
                total += weight*value
        if node.is_output():
            #total = self.activation(total)
            node.set_value(total)
        elif not node.is_output():
            total = self.activation(total)
            node.set_value(total)

    def forward_propogate(self):
        '''perform forward propogation to compute node outputs'''
        while True:
            nodes = self.get_nodes()
            N = len(nodes)
            for i in range(N):
                node = nodes[i]
                incoming_nodes = self.get_incoming(node)
                self.compute_node_value(node,incoming_nodes)
            if self.finished_computation_on_all_nodes():
                break
        nodes = self.get_nodes()
        for node in nodes:
            print (node,node.get_value())

    def set_train_facts(self,facts):
        '''sets training facts'''
        self.train_facts = facts

    def set_train_examples(self,examples):
        '''sets training examples'''
        self.train_examples = examples

    def add_node(self,node):
        '''add a node to the neural network'''
        self.nodes.append(node)

    def get_nodes(self):
        '''get all nodes in the network'''
        return self.nodes

    def make_input_node(self,input_node_desc):
        '''add input node(s) to network structure'''
        nodes = input_node_desc.split(":")[1].replace(" ","")
        nodes = nodes.split("|")
        for node in nodes:
            label = node.split("\"")[0][:-1]
            time_label = label[-1] #--> only max 9 length horizon allowed
            clause = node.split("\"")[1][:-1]+",v"+time_label+")"
            ip_node = Node(label,clause,ip=True)
            self.add_node(ip_node)

    def make_node(self,node_desc):
        '''add hidden node(s) to network structure'''
        nodes = node_desc.split(":")[1].replace(" ","")
        nodes = nodes.split("|")
        for node in nodes:
            label = node.split("\"")[0][:-1]
            time_label = label[-1] #--> only max 9 length horizon allowed
            clause = node.split("\"")[1][:-1]+",v"+time_label+")"
            h_node = Node(label,clause)
            self.add_node(h_node)

    def make_output_node(self,output_node_desc):
        '''add output node to network structure'''
        nodes = output_node_desc.split(":")[1].replace(" ","")
        nodes = nodes.split("|")
        for node in nodes:
            label = node.split("\"")[0][:-1]
            time_label = label[-1] #--> only max 9 length horizon allowed
            clause = node.split("\"")[1][:-1]+",v"+time_label+")"
            op_node = Node(label,clause,op=True)
            self.add_node(op_node)

    def make_connection(self,direction,weight):
        '''add connection between two nodes to network structure'''
        from_node_label = direction.split("-")[0]
        to_node_label = direction.split("-")[1]
        from_node,to_node = False,False
        for node in self.get_nodes():
            if not from_node:
                if node.get_label() == from_node_label:
                    from_node = node
            if not to_node:
                if node.get_label() == to_node_label:
                    to_node = node
        from_node.add_connection(to_node,weight)

    def make_connections(self,connection_desc):
        '''add all connections between nodes to network structure'''
        connections = connection_desc.split(":")[1].replace(" ","")
        connections = connections.split("|")
        for connection in connections:
            direction = connection.split(",")[0]
            weight = float(connection.split(",")[1])
            self.make_connection(direction,weight)

    def expand_nodes(self,nodes):
        '''expands recurrence of input nodes'''
        expanded_nodes = ""
        for node in nodes:
            node_label = node.split("(")[0]
            node_clause = node.split(node_label)[1]
            for i in range(self.horizon_length):
                expanded_node = node_label+str(i+1)+node_clause
                expanded_nodes += " | "+expanded_node
        return expanded_nodes[2:]

    def modify(self,desc_file):
        '''modifies desc_file by expanding recurrence'''
        modified_desc_file = []
        expanded_nodes = ""
        rec_node_label = None
        for desc in desc_file:
            if "rec_node" not in desc:
                continue
            rec_node = desc.replace(" ","").split(":")[1]
            rec_node_label = rec_node.split("(")[0]
            rec_node_clause = rec_node.split(rec_node_label)[1]
            for i in range(self.horizon_length):
                expanded_node = rec_node_label+str(i+1)+rec_node_clause
                expanded_nodes += " | "+expanded_node
        for desc in desc_file:
            if "input_node" in desc:
                nodes = desc.replace(" ","").split(":")[1].split("|")
                expanded = self.expand_nodes(nodes)
                modified_desc_file.append("input_node:"+expanded)
            if "node" in desc and "input_node" not in desc and "rec_node" not in desc and "output_node" not in desc:
                nodes = desc.replace(" ","").split(":")[1].split("|")
                expanded = self.expand_nodes(nodes)
                expanded += expanded_nodes
                modified_desc_file.append("node: "+expanded)
            if "rec_node" in desc:
                continue #already dealt with
            if "output_node" in desc:
                nodes = desc.replace(" ","").split(":")[1].split("|")
                expanded = self.expand_nodes(nodes)
                modified_desc_file.append("output_node:"+expanded)
            if "connect" in desc:
                expanded_connections = self.expand_connections(desc,rec_node_label)
                modified_desc_file.append("connect:"+expanded_connections)
                
        return modified_desc_file

    def expand_connections(self,desc,rec_node_label):
        '''expands recurrent connections'''
        expanded_connects = ""
        expanded_rec_connects = ""
        connects = desc.replace(" ","").split(":")[1].split("|")
        for i in range(self.horizon_length-1):
            expanded_rec_connects += " | "+rec_node_label+str(i+1)+"-"+rec_node_label+str(i+2)+",1"
        for connect in connects:
            edge = connect.split(",")[0]
            edge_weight = connect.split(",")[1]
            from_node = edge.split("-")[0]
            to_node = edge.split("-")[1]
            for i in range(self.horizon_length):
                expanded_from_node = from_node+str(i+1)
                expanded_to_node = to_node+str(i+1)
                expanded_connect = expanded_from_node+"-"+expanded_to_node+","+edge_weight
                expanded_connects += " | "+expanded_connect
        expanded_connects += expanded_rec_connects
        return expanded_connects[2:]
        

    def read_network_description(self,desc_file):
        '''reads description file
           adds input nodes, hidden nodes and output nodes
           expands recurrent node
           finally, adds connections between nodes with init weights
        '''
        desc_file = self.modify(desc_file)
        for desc in desc_file:
            if "input_node" in desc:
                self.make_input_node(desc)
            if "node" in desc and "input_node" not in desc and "output_node" not in desc:
                self.make_node(desc)
            if "output_node" in desc:
                self.make_output_node(desc)
            if "connect" in desc:
                self.make_connections(desc)

def main():
    '''main method'''
    file = []
    with open("desc.txt") as f:
        file = f.read().splitlines()
    facts,examples = [],[]
    with open("facts.txt") as f:
        facts = f.read().splitlines()
    with open("examples.txt") as f:
        examples = f.read().splitlines()

    net1 = RecRelNN(file)
    net1.learn(facts,examples)

main()
