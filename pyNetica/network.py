#from node import Node
from ctypes import c_int
from netica import Netica
from netica import NATURE_NODE, DECISION_NODE, UTILITY_NODE
from node import Node
import numpy as np
import sys

class Network(object):
    def __init__(self, name, lic=None, file=None):
        if file is None:
            self.name = name
            self.ntc = Netica()
            self.env = self.ntc.newenv(lic)
            self.ntc.initenv(self.env)
            self.net = self.ntc.newnet(name, self.env)
            self.nodes = []
        else:
            self.name = name
            self.ntc = Netica()
            self.env = self.ntc.newenv(lic)
            self.ntc.initenv(self.env)
            self.net = self.ntc.opennet(self.env, file)
            nodelist_p = self.ntc.getnetnodes(self.net)
            numnode = self.ntc.lengthnodelist(nodelist_p)
            self.nodes = []
            for i in range(numnode):
                nodei_p = self.ntc.nthnode(nodelist_p, i)
                nodename = self.ntc.getnodename(nodei_p)
                statename = 'init'; statenames = []; istate = 0
                while statename!='error':
                    statename = self.ntc.getnodestatename(nodei_p, istate)
                    statenames.append(statename)
                    istate += 1
                statenames = statenames[:-1]
                # default: no parents and continuous: therefore not the original nodes
                nodei = Node(nodename, parents=[], rvname='continuous')
                nodei.set_node_ptr(nodei_p)
                nodei.set_node_state_name(statenames)
                self.nodes.append(nodei)


    def add_nodes(self, nodes):
        for node in nodes:
            if (node.nodekind==NATURE_NODE) and (node.cpt is None or (node.cpt.size == 0)):
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


    def get_node_expectedutils(self, node):
        utils32 = self.ntc.getnodeexpectedutils(node.ptr)
        return utils32.astype('float')


    def enter_finding(self, node, evidence):
        if isinstance(evidence, int):
            stateindx = evidence
        elif isinstance(evidence, basestring):
            lowercaseStates = np.array([name.lower() for name in node.statenames])
            stateindx = np.where(lowercaseStates == evidence.lower())[0][0]
        stateindx = c_int(stateindx)
        self.ntc.enterfinding(node.ptr, stateindx)


    def retract_nodefindings(self, node):
        self.ntc.retractnodefindings(node.ptr)


    def retract_netfindings(self):
        self.ntc.retractnetfindings(self.net)


    def save_net(self, filename):
        self.ntc.savenet(self.env, self.net, filename)


    def set_node_kind(self, node, nodekind=NATURE_NODE):
        self.ntc.set_node_kind(node, nodekind)

    def get_node_funcstate(self, node, parentstate):
        return self.ntc.getnodefuncstate(node.ptr, parentstate)

    def find_nodenamed(self, nodename):
        """
        find node by node name
        """
        found = False
        for i,node in enumerate(self.nodes):
            if node.name == nodename:
                found =True
                break
        if found:
            return node
        else:
            return 'error'
