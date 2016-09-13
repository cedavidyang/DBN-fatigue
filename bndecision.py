from pyNetica import Node, Network
from pyNetica.netica import DECISION_NODE, UTILITY_NODE
from scipy import stats
import numpy as np
import itertools
import sys

from bnexample1_funcs import form2y

# random variables
rvX1 = stats.lognorm(1., scale=np.exp(0))
rvX3 = stats.lognorm(1., scale=np.exp(3*np.sqrt(2)))

# create nodes
weather = Node("Weather", parents=None, rvname='discrete')
forecast = Node("Forecast", parents=[weather], rvname='discrete')
umbrella = Node("Umbrella", parents=[forecast], rvname='discrete')
satisfy = Node("Satisfaction", parents=[weather, umbrella], rvname='deterministic')
umbrella.set_node_kind(DECISION_NODE)
satisfy.set_node_kind(UTILITY_NODE)

# assign CPT
# node weather
weathercpt = np.array([[0.7, 0.3]])
weather.assign_cpt(weathercpt, statenames=['sunshine', 'rain'])
# node forecast
forecastcpt = np.array([[0.7, 0.2, 0.1], [0.15, 0.25, 0.6]])
forecast.assign_cpt(forecastcpt, statenames=['sunny', 'cloudy', 'rainy'])
# node umbrella
umbrella.set_node_state_name(['take_umbrella', 'dont_take_umbrella'])
# node satisfy
def calculate_util(pstate):
    forecaststate,umbrellastate = pstate
    if forecaststate==0 and umbrellastate==0:
        return 20.
    elif forecaststate==0 and umbrellastate==1:
        return 100.
    elif forecaststate==1 and umbrellastate==0:
        return 70.
    elif forecaststate==1 and umbrellastate==1:
        return 0.
satisfy.assign_func(calculate_util)
# create new network
dbnet = Network("BNexample1")

# add nodes to network
dbnet.add_nodes([weather, forecast, umbrella, satisfy])
# add link: must before define nodes
dbnet.add_link()
# define nodes
dbnet.define_nodes()

# compile the net
dbnet.compile_net()
# enable autoupdate
dbnet.set_autoupdate()
# save the network
dbnet.save_net("BNdecision.dne")


# decision making
# type 1
dbnet.enter_finding(forecast, 0)
utils = dbnet.get_node_expectedutils(umbrella)
print 'If the forecast is sunny, expected utility of {} is {}, of {} is {}\n'.format(
        umbrella.statenames[0], utils[0],
        umbrella.statenames[1], utils[1])

dbnet.retract_netfindings()
dbnet.enter_finding(forecast, 1)
utils = dbnet.get_node_expectedutils(umbrella)
print 'If the forecast is cloudy, expected utility of {} is {}, of {} is {}\n'.format(
        umbrella.statenames[0], utils[0],
        umbrella.statenames[1], utils[1])

dbnet.retract_netfindings()
dbnet.enter_finding(forecast, 2)
utils = dbnet.get_node_expectedutils(umbrella)
print 'If the forecast is rainy, expected utility of {} is {}, of {} is {}\n'.format(
        umbrella.statenames[0], utils[0],
        umbrella.statenames[1], utils[1])

#type 2
dbnet.retract_netfindings()
dummy = dbnet.get_node_expectedutils(umbrella)
for fs in range(forecast.nstates()):
    decision = dbnet.get_node_funcstate(umbrella, [fs])
    print 'If the forecast is {}, the best decision is {}.\n'.format(
            forecast.statenames[fs], umbrella.statenames[decision])
