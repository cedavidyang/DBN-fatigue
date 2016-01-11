from pyNetica import Network, Node
import numpy as np

# create new net
netp = Network("ChestClinic")

# define nodes
visitAsia = Node("VisitAsia", parents=None, rvname='discrete')
smoking = Node("Smoking", parents=None, rvname='discrete')
tuberculosis = Node("Tuberculosis", parents=[visitAsia], rvname='discrete')
cancer = Node("Cancer", parents=[smoking], rvname='discrete')
tbOrCa = Node("TbOrCa", parents=[tuberculosis, cancer], rvname='discrete')
xRay = Node("XRay", parents=[tbOrCa], rvname='discrete')

# assign CPT
visitAsiaCpt = np.array([0.01, 0.99])[np.newaxis, :]
visitAsia.assign_cpt(visitAsiaCpt, statenames=['visit','no_visit'])
smokingCpt = np.array([0.50, 0.50])[np.newaxis, :]
smoking.assign_cpt(smokingCpt, statenames=['smoker','nonsmoker'])
tuberCpt = np.array([[0.05, 0.95], [0.01, 0.99]])
tuberculosis.assign_cpt(tuberCpt, statenames=['present','absent'])
cancerCpt = np.array([[0.10, 0.90], [0.01, 0.99]])
cancer.assign_cpt(cancerCpt, statenames=['present','absent'])
tbOrCaCpt = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
tbOrCa.assign_cpt(tbOrCaCpt, statenames=['true','false'])
xRayCpt = np.array([[0.98, 0.02], [0.05, 0.95]])
xRay.assign_cpt(xRayCpt, statenames=['abnormal','normal'])

# add node
netp.add_nodes([visitAsia])
netp.add_nodes([smoking])
netp.add_nodes([tuberculosis])
netp.add_nodes([cancer])
netp.add_nodes([tbOrCa, xRay])
# add link: must before define nodes
netp.add_link()
# define nodes
netp.define_nodes()

# compile the net
netp.compile_net()
# enable auto updating
netp.set_autoupdate()
# save the network
netp.save_net('ChestClinicNew.dne')

# prior belief
beliefs = netp.get_node_beliefs(tuberculosis)
belief = beliefs[0]
print 'The probability of tuberculosis is {:f}\n'.format(belief)

# posterior belief
netp.enter_finding(xRay, 0)
beliefs = netp.get_node_beliefs(tuberculosis)
belief = beliefs[0]
print 'Given an abnormal X-ray,'
print '         the probability of tuberculosis is {:f}\n'.format(belief)

netp.enter_finding(visitAsia, 0)
beliefs = netp.get_node_beliefs(tuberculosis)
belief = beliefs[0]
print 'Given an abnormal X-ray and a visit to Asia,'
print '         the probability of tuberculosis is {:f}\n'.format(belief)

netp.enter_finding(cancer, 0)
beliefs = netp.get_node_beliefs(tuberculosis)
belief = beliefs[0]
print 'Given abnormal X-ray, Asia visit, and lung cancer,'
print '         the probability of tuberculosis is {:f}\n'.format(belief)
