from math import exp
from Logic import Prover
from copy import deepcopy
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score

class Node(object):

    def __init__(self,label,clause,ip=False,op=False):
	'''constructor for node object'''

        self.label = label #node label == identifier
        self.clause = clause #node clause == either a single predicate or a clause
        self.ip = ip #whether input node or not
        self.op = op #whether output node or not
        self.delta = None #stores the gradients required during backprop
        if ip:
            self.value = 1 #output of the node
        else:
            self.value = None
        self.connections = [] #nodes that the node is connected to

    def set_delta(self,value):
	'''set gradient value'''
        self.delta = value

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
    
class RelNN(object):

    def __init__(self,description_file,learning_rate=0.01):
	'''constructor for Relational Neural Network'''
        self.nodes = []
        self.read_network_description(description_file)
        self.train_facts = []
        self.train_examples = []
        self.learning_rate = learning_rate
        self.test_facts = []
        self.test_examples = []

    def common_connection(self,s,node2):
        node2_connections = node2.get_connections()
        node2_connections = [str(n[0]) for n in node2_connections]
        for node in s:
            node_connections = node.get_connections()
            c = len([n for n in node_connections if str(n[0]) in node2_connections])
            if not c:
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
        clause_head = ""
        nodes = self.get_nodes()
        for node in nodes:
            rule = node.get_clause()
            rule_body = rule.split(";")
            for item in rule_body:
                item_name = item.split("(")[0]
                if item_name == literal_name:
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
                self.compute_ip_node_values(example)
                self.forward_propogate()
                self.backward_propogate(example)
        with open("NN_dot_file.dot","a") as df:
            df.write("digraph G {rankdir=LR;"+"\n")
            nodes = self.get_nodes()
            for node in nodes:
                if node.is_output():
                    continue
                cs = node.get_connections()
                for c in cs:
                    df.write("\""+node.get_label()+"("+node.get_clause()+")\"->\""+ c[0].get_label()+"("+c[0].get_clause()+")\"[label = "+str(c[1])+"]\n")
                    print (node.get_label(),c[0].get_label(),c[1])
            df.write("}")

    def compute_output_delta(self,example):
	'''computes output gradient (typically y - yhat)'''
        example_value = float(example.split(" ")[1])
        nodes = self.get_nodes()
        for node in nodes:
            if node.is_output():
                delta = example_value - node.get_value()
                node.set_delta(delta)

    def finished_all_deltas(self):
	'''checks if gradient at all nodes computed'''
        nodes = self.get_nodes()
        for node in nodes:
            if node.get_delta()==None:
                return False
        return True

    def compute_deltas(self,example):
	'''computes gradient at all nodes'''
        nodes = self.get_nodes()
        self.compute_output_delta(example)
        while True:
            for node in nodes:
                if node.is_output():
                    continue
                connections = node.get_connections()
                deltas_ready = True
                sum_of_deltas = 0
                for connection in connections:
                    if connection[0].get_delta() == None:
                        deltas_ready = False
                        break
                    weight = node.get_connection_weight(connection[0])
                    sum_of_deltas += weight*connection[0].get_delta()
                if not deltas_ready:
                    continue
                output = node.get_value()
                node.set_delta(output*(1-output)*sum_of_deltas)
            if self.finished_all_deltas():
                break

    def backward_propogate(self,example):
	'''performs backprop to learn weights'''
        self.compute_deltas(example)
        nodes = self.get_nodes()
        for node in nodes:
            if node.is_output():
                continue
            connections = node.get_connections()
            for connection in connections:
                c_node = connection[0]
                weight = node.get_connection_weight(c_node)
                weight += self.learning_rate*node.get_value()*c_node.get_delta()
                connection[1] = weight

    def set_test_facts(self,facts):
        '''sets testing facts'''
        self.test_facts = facts

    def set_test_examples(self,examples):
        self.test_examples = examples

    def set_train_facts(self,facts):
	'''sets training facts'''
        self.train_facts = facts

    def set_train_examples(self,examples):
	'''sets training examples'''
        self.train_examples = examples

    def activation(self,x):
	'''activation function at nodes'''
        return exp(x)/float(1+exp(x)) #sigmoid

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

    def finished_computation_on_all_nodes(self):
	'''checks if all node values have been computed'''
        for node in self.get_nodes():
            if node.get_value() == None:
                return False
        return True

    def test(self,facts,examples):
        '''compute value of examples by forward prop'''
        self.set_test_facts(facts)
        self.set_test_examples(examples)
        y_actual = []
        y_pred = []
        for example in self.test_examples:
            print ("example: ",example)
            y_actual.append(int(example.split(" ")[1]))
            self.compute_ip_node_values(example,test=True)
            self.forward_propogate()
            nodes = self.get_nodes()
            for node in nodes:
                if node.is_output():
                    node_value = node.get_value()
                    y_pred.append(int(node_value > 0.5))
                    print ("value: ",node_value)
        print ("accuracy: ",accuracy_score(y_actual,y_pred))
        print ("AUC-PR: ",average_precision_score(y_actual,y_pred))
        print ("AUC-ROC: ",roc_auc_score(y_actual,y_pred))

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

    def add_node(self,node):
	'''add a node to the neural node'''
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
            clause = node.split("\"")[1]
            ip_node = Node(label,clause,ip=True)
            self.add_node(ip_node)

    def make_node(self,node_desc):
	'''add hidden node(s) to network structure'''
        nodes = node_desc.split(":")[1].replace(" ","")
        nodes = nodes.split("|")
        for node in nodes:
            label = node.split("\"")[0][:-1]
            clause = node.split("\"")[1]
            h_node = Node(label,clause)
            self.add_node(h_node)

    def make_output_node(self,output_node_desc):
	'''add output node to network structure'''
        nodes = output_node_desc.split(":")[1].replace(" ","")
        nodes = nodes.split("|")
        for node in nodes:
            label = node.split("\"")[0][:-1]
            clause = node.split("\"")[1]
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

    def read_network_description(self,desc_file):
	'''read description file
	   adds input nodes, hidden nodes and output nodes
           also adds connections between nodes with init weights
	'''
        for desc in desc_file:
            if "input_node" in desc:
                self.make_input_node(desc)
            if "node" in desc and "input_node" not in desc and "output_node" not in desc:
                self.make_node(desc)
            if "output_node" in desc:
                self.make_output_node(desc)
            if "connect" in desc:
                self.make_connections(desc)

'''testing file reading'''
file = []
with open("desc.txt") as f:
    file = f.read().splitlines()
facts,examples,t_facts,t_examples = [],[],[],[]
with open("facts.txt") as f:
    facts = f.read().splitlines()
with open("examples.txt") as f:
    examples = f.read().splitlines()
with open("test_facts.txt") as f:
    t_facts = f.read().splitlines()
with open("test_examples.txt") as f:
    t_examples = f.read().splitlines()

'''testing neural network learning'''
net1 = RelNN(file)
net1.learn(facts,examples)
net1.test(t_facts,t_examples)
