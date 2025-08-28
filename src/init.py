"""
init.py

Starting script to run NetPyNE-based M1 model.

Usage:
    python init.py # Run simulation, optionally plot a raster

MPI usage:
    mpiexec -n 4 nrniv -python -mpi init.py

Contributors: salvadordura@gmail.com
"""

import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers
from netpyne import sim
import json
from netParams import netParams, cfg
from pathlib import Path
import defs

cwd = str(Path.cwd())

#------------------------------------------------------------------------------
# This for old batch, but before doing it uninstall batchtk
# cfg, netParams = sim.readCmdLineArgs(simConfigDefault=cwd+'/src/cfg.py', netParamsDefault=cwd+'/src/netParams.py')

sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations
sim.net.connectCells()            			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation
sim.setupRecording()

#------------------------------------------------------------------------------
# Simulation option 1: standard
sim.runSim()                              # run parallel Neuron simulation (calling func to modify mechs)
# # Simulation option 2: interval function to modify mechanism params
#TODO: Check that it works properly on CoreNEURON
# print(cfg.modifyMechs)
# sim.runSimWithIntervalFunc(cfg.transient+cfg.preTone, defs.modifyMechsFunc, funcArgs={'cfg': cfg})       # run parallel Neuron simulation (calling func to modify mechs)

sim.gatherData()                  			# gather spiking data and cell info from each node
# Gather/save data option 2: distributed saving across nodes
# sim.saveDataInNodes()
# sim.gatherDataFromFiles()

sim.simData.numSampledCellsPerLayer = cfg.numSampledCellsPerLayer
sim.simData.norm_layers = cfg.normLayers

sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#
sim.analysis.plotData()         			# plot spike raster etc

print('completed simulation...')

if sim.rank == 0:
    # netParams.save("{}/{}_params.json".format(cfg.saveFolder, cfg.simLabel))
    print('transmitting data...')
    inputs = cfg.get_mappings()
    # print(json.dumps({**inputs}))
    results = sim.analysis.popAvgRates(tranges=cfg.timeRanges, show=False) #TODO: Avoid printing firing rates

    sim.simData['popRates'] = results

    fitnessFuncArgs = {}
    pops = {}
    ## Exc pops
    Epops = ['IT2', 'IT4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6']
    Etune = {'target': 5, 'width': 5, 'min': 0.5}
    for pop in Epops:
        pops[pop] = Etune
    ## Inh pops
    Ipops = ['NGF1', 'PV2', 'SOM2', 'VIP2', 'NGF2',
             'PV4', 'SOM4', 'VIP4', 'NGF4',
             'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',
             'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',
             'PV6', 'SOM6', 'VIP6', 'NGF6']
    Itune = {'target': 10, 'width': 15, 'min': 0.25}
    for pop in Ipops:
        pops[pop] = Itune
    fitnessFuncArgs['pops'] = pops
    fitnessFuncArgs['maxFitness'] = 1000

    rateLoss = defs.rateFitnessFunc(sim.simData, **fitnessFuncArgs)
    results['loss'] = rateLoss
    out_json = json.dumps({**inputs, **results})

    print(out_json)
    sim.send(out_json)