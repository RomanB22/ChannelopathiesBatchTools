
"""
netParams.py 

High-level specifications for M1 network model using NetPyNE

Contributors: salvadordura@gmail.com
"""

from netpyne import specs
import pickle, json
from pathlib import Path
cwd = str(Path.cwd())
from cfg import cfg
import defs

cfg.update()
netParams = specs.NetParams()   # object of class NetParams to store the network parameters
netParams.version = cfg.version

#------------------------------------------------------------------------------
#
# NETWORK PARAMETERS
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# General network parameters
#------------------------------------------------------------------------------
netParams.scale = cfg.scale # Scale factor for number of cells
netParams.sizeX = cfg.sizeX # x-dimension (horizontal length) size in um
netParams.sizeY = cfg.sizeY # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = cfg.sizeZ # z-dimension (horizontal depth) size in um
netParams.shape = 'cylinder' # cylindrical (column-like) volume

#------------------------------------------------------------------------------
# General connectivity parameters
#------------------------------------------------------------------------------
netParams.scaleConnWeight = 1.0 # Connection weight scale factor (default if no model specified)
netParams.scaleConnWeightModels = {'HH_reduced': 1.0, 'HH_full': 1.0} #scale conn weight factor for each cell model
netParams.scaleConnWeightNetStims = 1.0 #0.5  # scale conn weight factor for NetStims
netParams.defaultThreshold = 0.0 # spike threshold, 10 mV is NetCon default, lower it for all cells
netParams.defaultDelay = 2.0 # default conn delay (ms)
netParams.propVelocity = 500.0 # propagation velocity (um/ms)
netParams.probLambda = 100.0  # length constant (lambda) for connection probability decay (um)
netParams.defineCellShapes = True  # convert stylized geoms to 3d points

#------------------------------------------------------------------------------
# Cell parameters
#------------------------------------------------------------------------------
cellModels = ['HH_reduced', 'HH_full']
excTypes = ['IT', 'CT', 'PT']
inhTypes = ['PV', 'SOM', 'VIP', 'NGF']

layer = cfg.normLayers
layer.update({'longTPO': [2.0,2.1], 'longTVL': [2.1,2.2], 'longS1': [2.2,2.3], 'longS2': [2.3,2.4], 
              'longcM1': [2.4,2.5], 'longM2': [2.5,2.6], 'longOC': [2.6,2.7]})  # normalized layer boundaries

netParams.correctBorder = {'threshold': [cfg.correctBorderThreshold, cfg.correctBorderThreshold, 
                                         cfg.correctBorderThreshold], 
                        'yborders': [layer['2'][0], layer['5A'][0], layer['6'][0], layer['6'][1]]}  # correct conn border effect

#------------------------------------------------------------------------------
## Load cell rules previously saved using netpyne format
cellParamLabels = cfg.cellParamLabels
loadCellParams = cellParamLabels
saveCellParams = False

for ruleLabel in loadCellParams:
    if ruleLabel!='PT5B_full':  # PT5B_full is loaded separately below. TODO: Only works for loading UCDavis model, not original
        netParams.loadCellParamsRule(label=ruleLabel, fileName='../cells/' + ruleLabel + '_cellParams.pkl')
    else:
        netParams.loadCellParamsRule(label='PT5B_full', fileName='../cells/Na12HH16HH_TF.json')

#------------------------------------------------------------------------------
# Specification of cell rules not previously loaded
# Includes importing from hoc template or python class, and setting additional params

#------------------------------------------------------------------------------
# Reduced cell model params (6-comp) 
reducedCells = {  # layer and cell type for reduced cell models
    'IT2_reduced':  {'layer': '2',  'cname': 'CSTR6', 'carg': 'BS1578'}, 
    'IT4_reduced':  {'layer': '4',  'cname': 'CSTR6', 'carg': 'BS1578'},
    'IT5A_reduced': {'layer': '5A', 'cname': 'CSTR6', 'carg': 'BS1579'},
    'IT5B_reduced': {'layer': '5B', 'cname': 'CSTR6', 'carg': 'BS1579'},
    'PT5B_reduced': {'layer': '5B', 'cname': 'SPI6',  'carg':  None},
    'IT6_reduced':  {'layer': '6',  'cname': 'CSTR6', 'carg': 'BS1579'},
    'CT6_reduced':  {'layer': '6',  'cname': 'CSTR6', 'carg': 'BS1578'}}

reducedSecList = {  # section Lists for reduced cell model
    'alldend':  ['Adend1', 'Adend2', 'Adend3', 'Bdend'],
    'spiny':    ['Adend1', 'Adend2', 'Adend3', 'Bdend'],
    'apicdend': ['Adend1', 'Adend2', 'Adend3'],
    'perisom':  ['soma']}

for label, p in reducedCells.items():  # create cell rules that were not loaded 
    if label not in loadCellParams:
        cellRule = defs.reducedCellModels(label, p, cwd, layer, cfg, reducedSecList, saveCellParams)
        netParams.cellParams[label] = cellRule
       
#------------------------------------------------------------------------------
## PT5B full cell model params (700+ comps)
if 'PT5B_full' not in loadCellParams:
    # cellRule = defs.PT5BFullModel(cfg, cwd, saveCellParams)
    cellRule = defs.PT5B_Tim(cfg, cwd, saveCellParams)
    netParams.cellParams['PT5B_full'] = cellRule

#------------------------------------------------------------------------------
## IT5A full cell model params (700+ comps)
if 'IT5A_full' not in loadCellParams:
    cellRule = defs.IT5AFullModel(cwd, saveCellParams, cfg, layer)
    netParams.cellParams['IT5A_full'] = cellRule

#------------------------------------------------------------------------------
## IT5B full cell model params (700+ comps) - not used
# if 'IT5B_full' not in loadCellParams:
#     cellRule = defs.IT5BFullModel(cwd, layer, saveCellParams)
#     netParams.cellParams['IT5B_full'] = cellRule

#------------------------------------------------------------------------------
## PV cell params (3-comp)
if 'PV_reduced' not in loadCellParams:
    cellRule = defs.PVReducedModel(cwd, cfg, saveCellParams)
    netParams.cellParams['PV_reduced'] = cellRule

#------------------------------------------------------------------------------
## SOM cell params (3-comp)
if 'SOM_reduced' not in loadCellParams:
    cellRule = defs.SOMReducedModel(cwd, cfg, saveCellParams)
    netParams.cellParams['SOM_reduced'] = cellRule

#------------------------------------------------------------------------------
## VIP cell params (5-comp)
if 'VIP_reduced' not in loadCellParams:
    cellRule = defs.VIPReducedModel(cwd, cfg, saveCellParams)
    netParams.cellParams['VIP_reduced'] = cellRule

#------------------------------------------------------------------------------
## NGF cell params (1-comp)
if 'NGF_reduced' not in loadCellParams:
    cellRule = defs.NGFReducedModel(cwd, cfg, saveCellParams)
    netParams.cellParams['NGF_reduced'] = cellRule

#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
## load densities
with open(cwd+'/cells/cellDensity.pkl', 'rb') as fileObj: density = pickle.load(fileObj)['density']
density = {k: [x * cfg.scaleDensity for x in v] for k,v in density.items()} # Scale densities 

defs.definePops(netParams, cfg, layer, density)

if cfg.singleCellPops:
    for pop in netParams.popParams.values(): pop['numCells'] = 1

#------------------------------------------------------------------------------
## Long-range input populations (VecStims)
if cfg.addLongConn:
    connLongData = defs.addLongConnections(cwd, netParams, cfg, layer)

#------------------------------------------------------------------------------
# Synaptic mechanism parameters
#------------------------------------------------------------------------------
netParams.synMechParams['NMDA'] = {'mod': 'MyExp2SynNMDABB', 'tau1NMDA': 15, 'tau2NMDA': 150, 'e': 0}
netParams.synMechParams['AMPA'] = {'mod':'MyExp2SynBB', 'tau1': 0.05, 'tau2': 5.3*cfg.AMPATau2Factor, 'e': 0}
netParams.synMechParams['GABAB'] = {'mod':'MyExp2SynBB', 'tau1': 3.5, 'tau2': 260.9, 'e': -93} 
netParams.synMechParams['GABAA'] = {'mod':'MyExp2SynBB', 'tau1': 0.07, 'tau2': 18.2, 'e': -80}
netParams.synMechParams['GABAA_VIP'] = {'mod':'MyExp2SynBB', 'tau1': 0.3, 'tau2': 6.4, 'e': -80}  # Pi et al 2013
netParams.synMechParams['GABAASlow'] = {'mod': 'MyExp2SynBB','tau1': 2, 'tau2': 100, 'e': -80}
netParams.synMechParams['GABAASlowSlow'] = {'mod': 'MyExp2SynBB', 'tau1': 200, 'tau2': 400, 'e': -80}

ESynMech = ['AMPA', 'NMDA']
SOMESynMech = ['GABAASlow','GABAB']
SOMISynMech = ['GABAASlow']
PVSynMech = ['GABAA']
VIPSynMech = ['GABAA_VIP']
NGFSynMech = ['GABAA', 'GABAB']

#------------------------------------------------------------------------------
# Long range input pulses
#------------------------------------------------------------------------------
if cfg.addPulses:
    defs.addStimPulses(cfg, netParams)

#------------------------------------------------------------------------------
# Current inputs (IClamp)
#------------------------------------------------------------------------------
if cfg.addIClamp:
    defs.addStimIclamp(cfg, netParams)

#------------------------------------------------------------------------------
# NetStim inputs
#------------------------------------------------------------------------------
if cfg.addNetStim:
    defs.addStimNetStim(cfg, netParams, ESynMech, SOMESynMech)

#------------------------------------------------------------------------------
# Local connectivity parameters
#------------------------------------------------------------------------------
with open(cwd+'/conn/conn.pkl', 'rb') as fileObj: connData = pickle.load(fileObj)
pmat = connData['pmat']
wmat = connData['wmat']
bins = connData['bins']

#------------------------------------------------------------------------------
## E -> E
if cfg.addConn and cfg.EEGain > 0.0:
    defs.defineEEConnections(bins, cfg, netParams, cellModels, pmat, wmat)

#------------------------------------------------------------------------------
## E -> I
if cfg.addConn and cfg.EIGain > 0.0:
    defs.defineEIConnections(excTypes, inhTypes, bins, cfg, netParams, pmat, wmat)

#------------------------------------------------------------------------------
## I -> E
if cfg.addConn and cfg.IEGain > 0.0:
    defs.defineIEConnections(excTypes, inhTypes, bins, cfg, netParams, pmat, PVSynMech, SOMESynMech, VIPSynMech, NGFSynMech)

#------------------------------------------------------------------------------
## I -> I
if cfg.addConn and cfg.IIGain > 0.0:
    defs.defineIIConnections(excTypes, inhTypes, bins, cfg, netParams, pmat, PVSynMech, SOMESynMech, VIPSynMech, NGFSynMech)

#------------------------------------------------------------------------------
# Long-range  connectivity parameters
#------------------------------------------------------------------------------
if cfg.addLongConn:
    defs.defineLongRangeConnections(connLongData, cfg, netParams, cellModels, ESynMech)

#------------------------------------------------------------------------------
# Subcellular connectivity (synaptic distributions)
#------------------------------------------------------------------------------         
if cfg.addSubConn:
    defs.defineSubcellularConnectivity(cwd, netParams, layer, ESynMech, SOMESynMech, VIPSynMech, NGFSynMech, inhTypes)

#------------------------------------------------------------------------------
# Description
#------------------------------------------------------------------------------
netParams.description = """ 
- M1 net, 6 layers, 7 cell types 
- NCD-based connectivity from  Weiler et al. 2008; Anderson et al. 2010; Kiritani et al. 2012; 
  Yamawaki & Shepherd 2015; Apicella et al. 2012
- Parametrized version based on Sam's code
- Updated cell models and mod files
- Added parametrized current inputs
- Fixed bug: prev was using cell models in /usr/site/nrniv/local/python/ instead of cells 
- Use 5 synsperconn for 5-comp cells (HH_reduced); and 1 for 1-comp cells (HH_simple)
- Fixed bug: made global h params separate for each cell model
- Fixed v_init for different cell models
- New IT cell with same geom as PT
- Cleaned cfg and moved background inputs here
- Set EIGain and IEGain for each inh cell type
- Added secLists for PT full
- Fixed reduced CT (wrong vinit and file)
- Added subcellular conn rules to distribute synapses
- PT full model soma centered at 0,0,0 
- Set cfg seeds here to ensure they get updated
- Added PVSOMGain and SOMPVGain
- PT subcellular distribution as a cfg param
- Cylindrical volume
- DefaultDelay (for local conns) = 2ms
- Added long range connections based on Yamawaki 2015a,b; Suter 2015; Hooks 2013; Meyer 2011
- Updated cell densities based on Tsai 2009; Lefort 2009; Katz 2011; Wall 2016; 
- Separated PV and SOM of L5A vs L5B
- Fixed bugs in local conn (PT, PV5, SOM5, L6)
- Added perisom secList including all sections 50um from soma
- Added subcellular conn rules (for both full and reduced models)
- Improved cell models, including PV and SOM fI curves
- Improved subcell conn rules based on data from Suter15, Hooks13 and others
- Adapted Bdend L of reduced cell models
- Made long pop rates a cfg param
- Set threshold to 0.0 mV
- Parametrized I->E/I layer weights
- Added missing subconn rules (IT6->PT; S1,S2,cM1->IT/CT; long->SOM/PV)
- Added threshold to weightNorm (PT threshold=10x)
- weightNorm threshold as a cfg parameter
- Separate PV->SOM, SOM->PV, SOM->SOM, PV->PV gains 
- Conn changes: reduced IT2->IT4, IT5B->CT6, IT5B,6->IT2,4,5A, IT2,4,5A,6->IT5B; increased CT->PV6+SOM6
- Parametrized PT ih gbar
- Added IFullGain parameter: I->E gain for full detailed cell models
- Replace PT ih with Migliore 2012
- Parametrized ihGbar, ihGbarBasal, dendNa, axonNa, axonRa, removeNa
- Replaced cfg list params with dicts
- Parametrized ihLkcBasal and AMPATau2Factor
- Fixed synMechWeightFactor
- Parametrized PT ih slope
- Added disynapticBias to I->E (Yamawaki&Shepherd,2015)
- Fixed E->CT bin 0.9-1.0
- Replaced GABAB with exp2syn and adapted synMech ratios
- Parametrized somaNa
- Added ynorm condition to NetStims
- Added option to play back recorded spikes into long-range inputs
- Fixed Bdend pt3d y location
- Added netParams.convertCellShapes = True to convert stylized geoms to 3d points
- New layer boundaries, cell densities, conn, FS+SOM L4 grouped with L2/3, low cortical input to L4
- v53: Increased exc->L4 based on Yamawaki 2015 fig 5
- v54: Moved from NetPyNE v0.7.9 to v0.9.1
- v55: Added VIP and NGF cells; updated I->E/I conn (cell-type and layer specific; interlaminar) and long->I
- v100: New numbering to separate from old model; dt=0.025; fixed L1 density
- v101: Parameterized long-range weights for each pop to use this in batch evol
- v102: Fixed bug in I->IT and I->VIP conn due to cell model; renamed _simple to _reduced 
- v103: Increased E->NGF/VIP and decreased default I->I; increased NGF weightNorm x1.5
- v104: Definition of defs.py file with functions encapsulating steps of model definition. Adaptation of the code to new batchtools format
"""
