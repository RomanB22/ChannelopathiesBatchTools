from netpyne import specs, sim
import json

# load data
file = 'crcns-ssc-3/ssc-3_lowrate2_spikes.json'
with open(file, 'r') as f: spks = json.load(f)

# Network parameters
netParams = specs.NetParams()  # object of class NetParams to store the network parameters

## Population parameters
netParams.popParams['S2'] = {'cellModel': 'VecStim', 'numCells': 1000, 'spkTimes': spks} #[[50, 100, 200, 300], [150, 240, 412, 320],[55, 105, 210, 330]] }


# Simulation options
cfg = specs.SimConfig()        # object of class cfg to store simulation configuration
cfg.duration = 6*1e3           # Duration of the simulation, in ms
cfg.dt = 0.025                # Internal integration timestep to use
cfg.verbose = False            # Show detailed messages 
cfg.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
cfg.recordStep = 1             # Step size in ms to save data (eg. V traces, LFP, etc)
cfg.filename = 'crcns-ssc-3/ssc-3_lowrate2_sim'  # Set file output name
cfg.savePickle = False         # Save params, network and sim output to pickle file
cfg.saveMat = False         # Save params, network and sim output to pickle file
cfg.saveJson=1
cfg.checkErrors=0

cfg.analysis['plotRaster'] = { 'saveFig': True, 'showFig': False, 'labels': 'overlay', 'popRates': True, 'orderInverse': True, 
							 'figSize': (12,10), 'lw': 0.3, 'markerSize':3, 'marker': '.', 'dpi': 300} 
cfg.analysis['plotSpikeHist'] = {'yaxis':'rate', 'binSize':5, 'graphType':'bar',
								'saveFig': True, 'showFig': False, 'figSize': (10,4), 'dpi': 300} 
cfg.analysis['plotSpikeStats'] = {'saveFig': True}
cfg.analysis['plotRatePSD'] = {'saveFig': True}


# Create network and run simulation
sim.createSimulateAnalyze(netParams = netParams, simConfig = cfg)    
