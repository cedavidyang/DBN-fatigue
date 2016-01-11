#from node import Node
from ctypes import c_int
from netica import Netica
import numpy as np
import sys

class Network(object):
    def __init__(self, name, lic=None):
        self.name = name
        self.ntc = Netica()
        self.env = self.ntc.newenv(lic)
        self.ntc.initenv(self.env)
        self.net = self.ntc.newnet(name, self.env)
        self.nodes = []


    def add_nodes(self, nodes):
        for node in nodes:
            if node.cpt is None or (node.cpt.size == 0):
                print "assign {} cpt first before add to network".format(node.name)
            #nodeptr = self.ntc.newnode(node.name, node.cpt.shape[0], self.net)
            #node.set_pointer(nodeptr)
            #node.set_net(self)
            node.add_to_net(self)
            self.nodes.append(node)


    def add_link(self):
        for node in self.nodes:
            if node.parents is not None:
                for parentNode in node.parents:
                    self.ntc.addlink(parent=parentNode.ptr, child=node.ptr)


    def define_nodes(self):
        """define nodes can only be implemented after all links are added
        """
        for node in self.nodes:
            node.define()


    def compile_net(self):
        self.ntc.compilenet(self.net)


    def set_autoupdate(self):
        self.ntc.setautoupdate(self.net)


    def get_node_beliefs(self, node):
        beliefs32 = self.ntc.getnodebeliefs(node.ptr)
        return beliefs32.astype('float')


    def enter_finding(self, node, evidence):
        if isinstance(evidence, int):
            stateindx = evidence
        elif isinstance(evidence, basestring):
            lowercaseStates = np.array([name.lower() for name in node.statenames])
            stateindx = np.where(lowercaseStates == evidence.lower())[0][0]
        stateindx = c_int(stateindx)
        self.ntc.enterfinding(node.ptr, stateindx)


    def save_net(self, filename):
        self.ntc.savenet(self.env, self.net, filename)