from math import exp
from Logic import Prover

class Node(object):

    def __init__(self,label,clause,ip=False,op=False):
        self.label = label
        self.clause = clause
        self.ip = ip
        self.op = op
        if ip:
            self.value = 1
        else:
            self.value = False
        self.connections = []

    def add_connection(self,node,weight):
        self.connections.append([node,weight])

    def get_connection_weight(self,node):
        for connection in self.get_connections():
            connected_node = connection[0]
            if node.get_label() == connected_node.get_label():
                weight = connection[1]
                return weight

    def get_connections(self):
        return self.connections

    def is_connection(self,node):
        for connection in self.get_connections():
            connected_node = connection[0]
            if node.get_label() == connected_node.get_label():
                return True
        return False

    def get_value(self):
        return self.value

    def set_value(self,value):
        self.value = value

    def get_label(self):
        return self.label

    def get_clause(self):
        return self.clause

    def is_input(self):
        return self.ip

    def is_output(self):
        return self.op
    
class RelNN(object):

    def __init__(self,description_file):
        self.nodes = []
        self.read_network_description(description_file)
        self.train_facts = []
        self.train_examples = []

    def compute_ip_node_values(self,example):
        predicate = example.split(" ")[0]
        facts = self.train_facts
        print (predicate)
        literal_name = predicate.split("(")[0]
        clause_head = ""
        for node in self.get_nodes():
            rule = node.get_clause()
            rule_body = rule.split(";")
            for item in rule_body:
                item_name = item.split("(")[0]
                if item_name == literal_name:
                    clause_head = item
        for node in self.get_nodes():
            if node.is_input():
                clause_body = node.get_clause().replace(";",",")
                clause = clause_head+":-"+clause_body
                if clause_body != "true":
                    truth = Prover.prove(facts,predicate,clause)
                    node.set_value(int(truth))
        for node in self.get_nodes():
            if node.is_input():
                print (node.get_clause(),node.get_value())
        exit()

    def learn(self,facts,examples):
        self.set_train_facts(facts)
        self.set_train_examples(examples)
        for example in self.train_examples:
            self.compute_ip_node_values(example)

    def set_train_facts(self,facts):
        self.train_facts = facts

    def set_train_examples(self,examples):
        self.train_examples = examples

    def activation(self,x):
        return exp(x)/float(1+exp(x))

    def get_incoming(self,node):
        incoming_nodes = []
        for n in self.get_nodes():
            if n.is_connection(node):
                incoming_nodes.append(n)
        return incoming_nodes

    def compute_node_value(self,node,incoming_nodes):
        total = 0
        if node.is_input():
            return
        for n in incoming_nodes:
            if not n.get_value():
                return
            else:
                value = n.get_value()
                weight = n.get_connection_weight(node)
                total += weight*value
        if node.is_output():
            node.set_value(total)
        else:
            node.set_value(self.activation(total))

    def finished_computation_on_all_nodes(self):
        for node in self.get_nodes():
            if not node.get_value():
                return False
        return True

    def forward_propogate(self,example):
        while True:
            for node in self.get_nodes():
                incoming_nodes = self.get_incoming(node)
                self.compute_node_value(node,incoming_nodes)
            if self.finished_computation_on_all_nodes():
                break
        for node in self.get_nodes():
            print (node.get_label(),node.get_value())

    def add_node(self,node):
        self.nodes.append(node)

    def get_nodes(self):
        return self.nodes

    def make_input_node(self,input_node_desc):
        nodes = input_node_desc.split(":")[1].replace(" ","")
        nodes = nodes.split("|")
        for node in nodes:
            label = node.split("\"")[0][:-1]
            clause = node.split("\"")[1]
            ip_node = Node(label,clause,ip=True)
            self.add_node(ip_node)

    def make_node(self,node_desc):
        nodes = node_desc.split(":")[1].replace(" ","")
        nodes = nodes.split("|")
        for node in nodes:
            label = node.split("\"")[0][:-1]
            clause = node.split("\"")[1]
            h_node = Node(label,clause)
            self.add_node(h_node)

    def make_output_node(self,output_node_desc):
        nodes = output_node_desc.split(":")[1].replace(" ","")
        nodes = nodes.split("|")
        for node in nodes:
            label = node.split("\"")[0][:-1]
            clause = node.split("\"")[1]
            op_node = Node(label,clause,op=True)
            self.add_node(op_node)

    def make_connection(self,direction,weight):
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
        connections = connection_desc.split(":")[1].replace(" ","")
        connections = connections.split("|")
        for connection in connections:
            direction = connection.split(",")[0]
            weight = float(connection.split(",")[1])
            self.make_connection(direction,weight)

    def read_network_description(self,desc_file):
        for desc in desc_file:
            if "input_node" in desc:
                self.make_input_node(desc)
            if "node" in desc and "input_node" not in desc and "output_node" not in desc:
                self.make_node(desc)
            if "output_node" in desc:
                self.make_output_node(desc)
            if "connect" in desc:
                self.make_connections(desc)

file = []
with open("desc.txt") as f:
    file = f.read().splitlines()
facts,examples = [],[]
with open("facts.txt") as f:
    facts = f.read().splitlines()
with open("examples.txt") as f:
    examples = f.read().splitlines()

net1 = RelNN(file)
net1.learn(facts,examples)


        
