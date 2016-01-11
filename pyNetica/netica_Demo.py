from netica import Netica
import numpy as np
 
# initialize class
ntc = Netica()
 
# create new environment
env = ntc.newenv()
# initialize environment
ntc.initenv(env)
# create new net
net_p = ntc.newnet('ChestClinic', env)
 
# define nodes
VisitAsia = ntc.newnode("VisitAsia", 2, net_p)
Tuberculosis = ntc.newnode ("Tuberculosis", 2, net_p)
Smoking =      ntc.newnode ("Smoking", 2, net_p)
Cancer =       ntc.newnode ("Cancer", 2, net_p)
TbOrCa =       ntc.newnode ("TbOrCa", 2, net_p)
XRay =         ntc.newnode ("XRay", 2, net_p)

# set node names
ntc.setnodestatenames(VisitAsia,   "visit,   no_visit");
ntc.setnodestatenames(Tuberculosis,"present, absent");
ntc.setnodestatenames(Smoking,     "smoker,  nonsmoker");
ntc.setnodestatenames(Cancer,      "present, absent");
ntc.setnodestatenames(TbOrCa,      "true,    false");
ntc.setnodestatenames(XRay,        "abnormal,normal");

# define links
ntc.addlink(parent=VisitAsia,    child=Tuberculosis)
ntc.addlink(parent=Smoking,      child=Cancer)
ntc.addlink(parent=Tuberculosis, child=TbOrCa)
ntc.addlink(parent=Cancer,       child=TbOrCa)
ntc.addlink(parent=TbOrCa,       child=XRay)

# set node probs

parent_states = np.empty((1,),dtype='int32')
probs = np.array([0.01, 0.99], dtype='float32')
ntc.setnodeprobs (VisitAsia, parent_states, probs)

parent_states = np.array([0],dtype='int32')
probs = np.array([0.05, 0.95], dtype='float32')
ntc.setnodeprobs (Tuberculosis, parent_states, probs)
parent_states = np.array([1],dtype='int32')
probs = np.array([0.01, 0.99], dtype='float32')
ntc.setnodeprobs (Tuberculosis, parent_states, probs)

parent_states = np.empty((1,),dtype='int32')
probs = np.array([0.50, 0.50], dtype='float32')
ntc.setnodeprobs (Smoking, parent_states, probs)

parent_states = np.array([0],dtype='int32')
probs = np.array([0.10, 0.90], dtype='float32')
ntc.setnodeprobs (Cancer, parent_states, probs)
parent_states = np.array([1],dtype='int32')
probs = np.array([0.01, 0.99], dtype='float32')
ntc.setnodeprobs (Cancer, parent_states, probs)

parent_states = np.array([0, 0], dtype='int32')
probs = np.array([1.0, 0.0], dtype='float32')
ntc.setnodeprobs (TbOrCa, parent_states, probs)
parent_states = np.array([0, 1], dtype='int32')
probs = np.array([1.0, 0.0], dtype='float32')
ntc.setnodeprobs (TbOrCa, parent_states, probs)
parent_states = np.array([1, 0],dtype='int32')
probs = np.array([1.0, 0.0], dtype='float32')
ntc.setnodeprobs (TbOrCa, parent_states, probs)
parent_states = np.array([1, 1],dtype='int32')
probs = np.array([0.0, 1.0], dtype='float32')
ntc.setnodeprobs (TbOrCa, parent_states, probs)

parent_states = np.array([0],dtype='int32')
probs = np.array([0.98, 0.02], dtype='float32')
ntc.setnodeprobs (XRay, parent_states, probs)
parent_states = np.array([1],dtype='int32')
probs = np.array([0.05, 0.95], dtype='float32')
ntc.setnodeprobs (XRay, parent_states, probs)

# compile the net
ntc.compilenet(net_p)
# enable auto updating
ntc.setautoupdate(net_p)

# prior belief
beliefs = ntc.getnodebeliefs(Tuberculosis)
belief = beliefs[0]
print 'The probability of tuberculosis is {:f}\n'.format(belief)

# posterior belief
ntc.enterfinding (XRay, 0)
beliefs = ntc.getnodebeliefs(Tuberculosis)
belief = beliefs[0]
print 'Given an abnormal X-ray,'
print '         the probability of tuberculosis is {:f}\n'.format(belief)

ntc.enterfinding (VisitAsia, 0)
beliefs = ntc.getnodebeliefs(Tuberculosis)
belief = beliefs[0]
print 'Given an abnormal X-ray and a visit to Asia,'
print '         the probability of tuberculosis is {:f}\n'.format(belief)

ntc.enterfinding (Cancer, 0)
beliefs = ntc.getnodebeliefs(Tuberculosis)
belief = beliefs[0]
print 'Given abnormal X-ray, Asia visit, and lung cancer,'
print '         the probability of tuberculosis is {:f}\n'.format(belief)

# save net
ntc.savenet(env, net_p, 'ChestClinic.dne')
