"""
defs.py

Definition of the cells and auxiliar functions used in the model

Contributors: romanbaravalle@gmail.com
"""
from netpyne import specs
import gc
import random
import csv
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Union
import math

#------------------------------------------------------------------------------
## Function to calculate the fitness according to required rate
def rateFitnessFunc(simData, extraConds=False, **kwargs):
    import numpy as np
    pops = kwargs['pops']
    maxFitness = kwargs['maxFitness']

    factor=1
    # Add extra conditions to the fitness. It 'breaks' the fitness function
    if extraConds:
        # check I > E in each layer
        condsIE_L23 = (simData['popRates']['PV2'] > simData['popRates']['IT2']) and (simData['popRates']['SOM2'] > simData['popRates']['IT2'])
        condsIE_L5A = (simData['popRates']['PV5A'] > simData['popRates']['IT5A']) and (simData['popRates']['SOM5A'] > simData['popRates']['IT5A'])
        condsIE_L5B = (simData['popRates']['PV5B'] > simData['popRates']['IT5B']) and (simData['popRates']['SOM5B'] > simData['popRates']['IT5B'])
        condsIE_L6 = (simData['popRates']['PV6'] > simData['popRates']['IT6']) and (simData['popRates']['SOM6'] > simData['popRates']['IT6'])
        # check E L5 > L6 > L2
        condEE562_0 = (simData['popRates']['IT5A']+simData['popRates']['IT5B']+simData['popRates']['PT5B'])/3 > (simData['popRates']['IT6']+simData['popRates']['CT6'])/2
        condEE562_1 = (simData['popRates']['IT6']+simData['popRates']['CT6'])/2 > simData['popRates']['IT2']
        # check PV > SOM in each layer
        condsPVSOM_L23 = (simData['popRates']['PV2'] > simData['popRates']['SOM2'])
        condsPVSOM_L5A = (simData['popRates']['PV5A'] > simData['popRates']['SOM5A'])
        condsPVSOM_L5B = (simData['popRates']['PV5B'] > simData['popRates']['SOM5B'])
        condsPVSOM_L6 = (simData['popRates']['PV6'] > simData['popRates']['SOM6'])

        conds = [condsIE_L23, condsIE_L5A, condsIE_L5B, condsIE_L6, condEE562_0, condEE562_1, condsPVSOM_L23, condsPVSOM_L5A, condsPVSOM_L5B, condsPVSOM_L6]

        if not all(conds): factor = 1.5
        
    popFitness = [min(np.exp(factor*abs(v['target'] - simData['popRates'][k])/v['width']), maxFitness) 
                if simData['popRates'][k] > v['min'] else maxFitness for k,v in pops.items()]
    fitness = np.mean(popFitness)

    popInfo = '; '.join(['%s rate=%.1f fit=%1.f'%(p, simData['popRates'][p], popFitness[i]) for i,p in enumerate(pops)])
    print('  '+popInfo)
    return fitness
#------------------------------------------------------------------------------
## Function to modify cell params during sim (e.g. modify PT ih)
def modifyMechsFunc(simTime, cfg):
    from netpyne import sim

    t = simTime

    cellType = cfg.modifyMechs['cellType']
    mech = cfg.modifyMechs['mech']
    prop = cfg.modifyMechs['property']
    newFactor = cfg.modifyMechs['newFactor']
    origFactor = cfg.modifyMechs['origFactor']
    factor = newFactor / origFactor
    change = False

    if cfg.modifyMechs['endTime']-1.0 <= t <= cfg.modifyMechs['endTime']+1.0:
        factor = origFactor / newFactor if abs(newFactor) > 0.0 else origFactor
        change = True

    elif t >= cfg.modifyMechs['startTime']-1.0 <= t <= cfg.modifyMechs['startTime']+1.0:
        factor = newFactor / origFactor if abs(origFactor) > 0.0 else newFactor
        change = True

    if change:
        print('   Modifying %s %s %s by a factor of %f' % (cellType, mech, prop, factor))
        for cell in sim.net.cells:
            if 'cellType' in cell.tags and cell.tags['cellType'] == cellType:
                for secName, sec in cell.secs.items():
                    if mech in sec['mechs'] and prop in sec['mechs'][mech]:
                        # modify python
                        sec['mechs'][mech][prop] = [g * factor for g in sec['mechs'][mech][prop]] if isinstance(sec['mechs'][mech][prop], list) else sec['mechs'][mech][prop] * factor

                        # modify neuron
                        for iseg, seg in enumerate(sec['hObj']):  # set mech params for each segment
                            if sim.cfg.verbose: print('   Modifying %s %s %s by a factor of %f' % (secName, mech, prop, factor))
                            setattr(getattr(seg, mech), prop, getattr(getattr(seg, mech), prop) * factor)
    return None

def reducedCellModels(label, p, cwd, layer, cfg, reducedSecList, saveCellParams):
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label=label, conds={'cellType': label[0:2], 'cellModel': 'HH_reduced', 'ynorm': layer[p['layer']]},
    fileName=cwd+'/cells/'+p['cname']+'.py', cellName=p['cname'], cellArgs={'params': p['carg']} if p['carg'] else None)
    dendL = (layer[p['layer']][0]+(layer[p['layer']][1]-layer[p['layer']][0])/2.0) * cfg.sizeY  # adapt dend L based on layer
    for secName in ['Adend1', 'Adend2', 'Adend3', 'Bdend']: cellRule['secs'][secName]['geom']['L'] = dendL / 3.0  # update dend L
    for k,v in reducedSecList.items(): cellRule['secLists'][k] = v  # add secLists
    netParamsAux.addCellParamsWeightNorm(label, cwd+'/conn/'+label+'_weightNorm.pkl', threshold=cfg.weightNormThreshold)  # add weightNorm

    # set 3d points
    offset, prevL = 0, 0
    somaL = netParamsAux.cellParams[label]['secs']['soma']['geom']['L']
    for secName, sec in netParamsAux.cellParams[label]['secs'].items():
        sec['geom']['pt3d'] = []
        if secName in ['soma', 'Adend1', 'Adend2', 'Adend3']:  # set 3d geom of soma and Adends
            sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
            prevL = float(prevL + sec['geom']['L'])
            sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
        if secName in ['Bdend']:  # set 3d geom of Bdend
            sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
            sec['geom']['pt3d'].append([offset+sec['geom']['L'], somaL, 0, sec['geom']['diam']])        
        if secName in ['axon']:  # set 3d geom of axon
            sec['geom']['pt3d'].append([offset+0, 0, 0, sec['geom']['diam']])
            sec['geom']['pt3d'].append([offset+0, -sec['geom']['L'], 0, sec['geom']['diam']])   

    if saveCellParams: netParamsAux.saveCellParamsRule(label=label, fileName=cwd+'/cells/'+label+'_cellParams.pkl')
    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def csv_to_dict(filepath):
    result = {}
    with open(filepath, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        key_field = fieldnames[0]  # Use the first column as key
        for row in reader:
            key = row[key_field]
            value = {k: v for k, v in row.items() if k != key_field}
            result[key] = value
    return result

def PT5B_Tim(cfg, cwd, saveCellParams):
    netParamsAux = specs.NetParams()
    #Load CSV with Mutant Params
    if cfg.loadmutantParams == True:
        print("Loading mutant params: ", cfg.variant)
    else:
        cfg.variant = 'WT'

    variants = csv_to_dict('./cells/Neuron_HH_Adult-main/MutantParameters_updated_062725.csv')
    sorted_variant = dict(sorted(variants[cfg.variant].items()))
    for key, value in sorted_variant.items():
        sorted_variant[key] = float(value)
    with open('./cells/Neuron_HH_Adult-main/Neuron_Model_12HH16HH/params/na12annaTFHH2mut.txt', 'w') as f:
        json.dump(sorted_variant, f)
    ###
    netParamsAux.importCellParams(label='PT5B_full', fileName='./cells/Neuron_HH_Adult-main/Na12HMMModel_TF.py',
                               cellName='Na12Model_TF')

    # rename soma to conform to netpyne standard
    netParamsAux.renameCellParamsSec(label='PT5B_full', oldSec='soma_0', newSec='soma')

    # set variable so easier to work with below
    cellRule = netParamsAux.cellParams['PT5B_full']

    # set the spike generation location to the axon (default in NEURON is the soma)
    cellRule['secs']['axon_0']['spikeGenLoc'] = 0.5

    # add pt3d for axon sections so SecList does not break
    cellRule['secs']['axon_0']['geom']['pt3d'] = [[-25.435224533081055, 34.14994812011719, 0, 1.6440753936767578],
                                                  [-25.065839767456055, 34.10675811767578, 0, 1.6440753936767578],
                                                  [-24.327072143554688, 34.02037811279297, 0, 1.6440753936767578],
                                                  [-23.588302612304688, 33.933998107910156, 0, 1.6440753936767578],
                                                  [-22.849533081054688, 33.847618103027344, 0, 1.6440753936767578],
                                                  [-22.11076545715332, 33.7612419128418, 0, 1.6440753936767578],
                                                  [-21.37199592590332, 33.674861907958984, 0, 1.6440753936767578],
                                                  [-20.63322639465332, 33.58848190307617, 0, 1.6440753936767578],
                                                  [-19.894458770751953, 33.50210189819336, 0, 1.6440753936767578],
                                                  [-19.155689239501953, 33.41572189331055, 0, 1.6440753936767578],
                                                  [-18.416919708251953, 33.329341888427734, 0, 1.6440753936767578],
                                                  [-17.678152084350586, 33.24296188354492, 0, 1.6440753936767578],
                                                  [-16.939382553100586, 33.15658187866211, 0, 1.6440753936767578],
                                                  [-16.200613021850586, 33.0702018737793, 0, 1.6440753936767578],
                                                  [-15.461844444274902, 32.98382568359375, 0, 1.6440753936767578],
                                                  [-14.723075866699219, 32.89744567871094, 0, 1.6440753936767578],
                                                  [-13.984307289123535, 32.811065673828125, 0, 1.6440753936767578],
                                                  [-13.245537757873535, 32.72468566894531, 0, 1.6440753936767578],
                                                  [-12.506769180297852, 32.6383056640625, 0, 1.6440753936767578],
                                                  [-11.768000602722168, 32.55192565917969, 0, 1.6440753936767578],
                                                  [-11.029231071472168, 32.465545654296875, 0, 1.6440753936767578],
                                                  [-10.290462493896484, 32.37916564941406, 0, 1.6440753936767578],
                                                  [-9.5516939163208, 32.29278564453125, 0, 1.6440753936767578],
                                                  [-8.8129243850708, 32.2064094543457, 0, 1.6440753936767578],
                                                  [-8.074155807495117, 32.12002944946289, 0, 1.6440753936767578],
                                                  [-7.335386753082275, 32.03364944458008, 0, 1.6440753936767578],
                                                  [-6.596618175506592, 31.947269439697266, 0, 1.6440753936767578],
                                                  [-5.85784912109375, 31.860889434814453, 0, 1.6440753936767578],
                                                  [-5.119080066680908, 31.77450942993164, 0, 1.6440753936767578],
                                                  [-4.380311489105225, 31.68813133239746, 0, 1.6440753936767578],
                                                  [-3.641542434692383, 31.60175132751465, 0, 1.6440753936767578],
                                                  [-2.90277361869812, 31.515371322631836, 0, 1.6440753936767578],
                                                  [-2.1640045642852783, 31.428991317749023, 0, 1.6440753936767578],
                                                  [-1.4252357482910156, 31.342613220214844, 0, 1.6440753936767578],
                                                  [-0.6864668726921082, 31.25623321533203, 0, 1.6440753936767578],
                                                  [0.05230199918150902, 31.16985321044922, 0, 1.6440753936767578],
                                                  [0.7910708785057068, 31.083473205566406, 0, 1.6440753936767578],
                                                  [1.5298397541046143, 30.997093200683594, 0, 1.6440753936767578],
                                                  [2.268608570098877, 30.910715103149414, 0, 1.6440753936767578],
                                                  [3.0073776245117188, 30.8243350982666, 0, 1.6440753936767578],
                                                  [3.7461464405059814, 30.73795509338379, 0, 1.6440753936767578],
                                                  [4.484915256500244, 30.651575088500977, 0, 1.6440753936767578],
                                                  [5.223684310913086, 30.565196990966797, 0, 1.6440753936767578],
                                                  [5.9624528884887695, 30.478816986083984, 0, 1.6440753936767578],
                                                  [6.701221942901611, 30.392436981201172, 0, 1.6440753936767578],
                                                  [7.439990997314453, 30.30605697631836, 0, 1.6440753936767578],
                                                  [8.178759574890137, 30.21967887878418, 0, 1.6440753936767578],
                                                  [8.91752815246582, 30.133298873901367, 0, 1.6440753936767578],
                                                  [9.65629768371582, 30.046918869018555, 0, 1.6440753936767578],
                                                  [10.395066261291504, 29.960538864135742, 0, 1.6440753936767578],
                                                  [11.133834838867188, 29.87415885925293, 0, 1.6440753936767578],
                                                  [11.872604370117188, 29.78778076171875, 0, 1.6440753936767578],
                                                  [12.611372947692871, 29.701400756835938, 0, 1.6440753936767578],
                                                  [13.350141525268555, 29.615020751953125, 0, 1.6440753936767578],
                                                  [14.088911056518555, 29.528640747070312, 0, 1.6440753936767578],
                                                  [14.827679634094238, 29.442262649536133, 0, 1.6440753936767578],
                                                  [15.566448211669922, 29.35588264465332, 0, 1.6440753936767578],
                                                  [16.305217742919922, 29.269502639770508, 0, 1.6440753936767578],
                                                  [17.043987274169922, 29.183122634887695, 0, 1.6440753936767578],
                                                  [17.78275489807129, 29.096744537353516, 0, 1.6440753936767578],
                                                  [18.52152442932129, 29.010364532470703, 0, 1.6440753936767578],
                                                  [19.260292053222656, 28.92398452758789, 0, 1.6440753936767578],
                                                  [19.999061584472656, 28.837604522705078, 0, 1.6440753936767578],
                                                  [20.737831115722656, 28.751224517822266, 0, 1.6440753936767578],
                                                  [21.476598739624023, 28.664846420288086, 0, 1.6440753936767578],
                                                  [22.215368270874023, 28.578466415405273, 0, 1.6440753936767578],
                                                  [22.954137802124023, 28.49208641052246, 0, 1.6440753936767578],
                                                  [23.69290542602539, 28.40570640563965, 0, 1.6440753936767578],
                                                  [24.43167495727539, 28.31932830810547, 0, 1.6440753936767578],
                                                  [25.17044448852539, 28.232948303222656, 0, 1.6440753936767578],
                                                  [25.909212112426758, 28.146568298339844, 0, 1.6440753936767578],
                                                  [26.647981643676758, 28.06018829345703, 0, 1.6440753936767578],
                                                  [27.386751174926758, 27.97381019592285, 0, 1.6440753936767578],
                                                  [28.125518798828125, 27.88743019104004, 0, 1.6440753936767578],
                                                  [28.864288330078125, 27.801050186157227, 0, 1.6440753936767578],
                                                  [29.603057861328125, 27.714670181274414, 0, 1.6440753936767578],
                                                  [30.341825485229492, 27.6282901763916, 0, 1.6440753936767578],
                                                  [31.080595016479492, 27.541912078857422, 0, 1.6440753936767578],
                                                  [31.819364547729492, 27.45553207397461, 0, 1.6440753936767578],
                                                  [32.55813217163086, 27.369152069091797, 0, 1.6440753936767578],
                                                  [33.29690170288086, 27.282772064208984, 0, 1.6440753936767578],
                                                  [34.03567123413086, 27.196393966674805, 0, 1.6440753936767578],
                                                  [34.77444076538086, 27.110013961791992, 0, 1.6440753936767578],
                                                  [35.51321029663086, 27.02363395690918, 0, 1.6440753936767578],
                                                  [36.251976013183594, 26.937253952026367, 0, 1.6440753936767578],
                                                  [36.990745544433594, 26.850875854492188, 0, 1.6440753936767578],
                                                  [37.729515075683594, 26.764495849609375, 0, 1.6440753936767578],
                                                  [38.468284606933594, 26.678115844726562, 0, 1.6440753936767578],
                                                  [39.207054138183594, 26.59173583984375, 0, 1.6440753936767578],
                                                  [39.945823669433594, 26.505355834960938, 0, 1.6440753936767578],
                                                  [40.68458938598633, 26.418977737426758, 0, 1.6440753936767578],
                                                  [41.42335891723633, 26.332597732543945, 0, 1.6440753936767578],
                                                  [42.16212844848633, 26.246217727661133, 0, 1.6440753936767578],
                                                  [42.90089797973633, 26.15983772277832, 0, 1.6440753936767578],
                                                  [43.63966751098633, 26.07345962524414, 0, 1.6440753936767578],
                                                  [44.37843322753906, 25.987079620361328, 0, 1.6440753936767578],
                                                  [45.11720275878906, 25.900699615478516, 0, 1.6440753936767578],
                                                  [45.85597229003906, 25.814319610595703, 0, 1.6440753936767578],
                                                  [46.59474182128906, 25.727941513061523, 0, 1.6440753936767578],
                                                  [47.33351135253906, 25.64156150817871, 0, 1.6440753936767578],
                                                  [48.07228088378906, 25.5551815032959, 0, 1.6440753936767578],
                                                  [48.8110466003418, 25.468801498413086, 0, 1.6440753936767578],
                                                  [49.5498161315918, 25.382421493530273, 0, 1.6440753936767578],
                                                  [50.2885856628418, 25.296043395996094, 0, 1.6440753936767578],
                                                  [51.0273551940918, 25.20966339111328, 0, 1.6440753936767578],
                                                  [51.7661247253418, 25.12328338623047, 0, 1.6440753936767578],
                                                  [52.5048942565918, 25.036903381347656, 0, 1.6440753936767578],
                                                  [53.24365997314453, 24.950525283813477, 0, 1.6440753936767578],
                                                  [53.98242950439453, 24.864145278930664, 0, 1.6440753936767578],
                                                  [54.72119903564453, 24.77776527404785, 0, 1.6440753936767578],
                                                  [55.45996856689453, 24.69138526916504, 0, 1.6440753936767578],
                                                  [56.19873809814453, 24.60500717163086, 0, 1.6440753936767578],
                                                  [56.93750762939453, 24.518627166748047, 0, 1.6440753936767578],
                                                  [57.676273345947266, 24.432247161865234, 0, 1.6440753936767578],
                                                  [58.415042877197266, 24.345867156982422, 0, 1.6440753936767578],
                                                  [59.153812408447266, 24.25948715209961, 0, 1.6440753936767578],
                                                  [59.892581939697266, 24.17310905456543, 0, 1.6440753936767578],
                                                  [60.631351470947266, 24.086729049682617, 0, 1.6440753936767578],
                                                  [61.370121002197266, 24.000349044799805, 0, 1.6440753936767578],
                                                  [62.10888671875, 23.913969039916992, 0, 1.6440753936767578],
                                                  [62.84765625, 23.827590942382812, 0, 1.6440753936767578],
                                                  [63.58642578125, 23.7412109375, 0, 1.6440753936767578],
                                                  [63.955810546875, 23.698020935058594, 0, 1.6440753936767578]]
    cellRule['secs']['axon_1']['geom']['pt3d'] = [[63.955810546875, 23.698020935058594, 0, 1.6440753936767578],
                                                  [65.31021881103516, 23.539657592773438, 0, 1.6440753936767578],
                                                  [68.01904296875, 23.222932815551758, 0, 1.6440753936767578],
                                                  [70.72785949707031, 22.906208038330078, 0, 1.6440753936767578],
                                                  [73.43667602539062, 22.5894832611084, 0, 1.6440753936767578],
                                                  [76.14550018310547, 22.27275848388672, 0, 1.6440753936767578],
                                                  [78.85431671142578, 21.956031799316406, 0, 1.6440753936767578],
                                                  [81.5631332397461, 21.639307022094727, 0, 1.6440753936767578],
                                                  [84.27195739746094, 21.322582244873047, 0, 1.6440753936767578],
                                                  [86.98077392578125, 21.005857467651367, 0, 1.6440753936767578],
                                                  [89.68959045410156, 20.689132690429688, 0, 1.6440753936767578],
                                                  [92.3984146118164, 20.372407913208008, 0, 1.6440753936767578],
                                                  [93.75282287597656, 20.21404457092285, 0, 1.6440753936767578]]

    # define cell conds
    cellRule['conds'] = {'cellModel': 'HH_full', 'cellType': 'PT'}

    # clean secLists from Tim's code
    cellRule['secLists'] = {}

    # create lists useful to define location of synapses
    nonSpiny = ['apic_0', 'apic_1']  # TODO: Where this comes from?

    netParamsAux.addCellParamsSecList(label='PT5B_full', secListName='perisom',
                                   somaDist=[0, 50])  # sections within 50 um of soma
    netParamsAux.addCellParamsSecList(label='PT5B_full', secListName='below_soma',
                                   somaDistY=[-600, 0])  # sections within 0-300 um of soma
    cellRule['secLists']['alldend'] = [sec for sec in cellRule.secs if ('dend' in sec or 'apic' in sec)]  # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule.secs if ('apic' in sec)]  # apical
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in nonSpiny]

    for sec in nonSpiny:  # N.B. apic_1 not in `perisom` . `apic_0` and `apic_114` are
        if sec in cellRule['secLists']['perisom']:  # fixed logic
            cellRule['secLists']['perisom'].remove(sec)

    if cfg.heterozygous:
        for secName in cellRule['secs']:
            for mechName, mech in cellRule['secs'][secName]['mechs'].items():
                if mechName in ['na12mut']:
                    mech['gbar'] = [g * 0. for g in mech['gbar']] if isinstance(mech['gbar'], list) else mech['gbar'] * 0.

    # for secName in cellRule['secs']:
    #     print(secName, cellRule['secs'][secName]['mechs']['Ih']['gIhbar'])

    # Adapt ih params based on cfg param
    for secName in cellRule['secs']:
        for mechName, mech in cellRule['secs'][secName]['mechs'].items():
            if mechName in ['Ih']:
                if secName not in ['axon_0', 'axon_1']:
                    mech['gIhbar'] = [g * cfg.ihGbar for g in mech['gIhbar']] if isinstance(mech['gIhbar'], list) else mech['gIhbar'] * cfg.ihGbar
                    if secName.startswith('dend'):
                        mech['gIhbar'] *= cfg.ihGbarBasal  # modify ih conductance in soma+basal dendrites

    # Decrease dendritic Na
    for secName in cellRule['secs']:
        if secName.startswith('apic'):
            cellRule['secs'][secName]['mechs']['na12']['gbar'] *= cfg.dendNa
            cellRule['secs'][secName]['mechs']['na12mut']['gbar'] *= cfg.dendNa

    # set weight normalization
    netParamsAux.addCellParamsWeightNorm('PT5B_full', './conn/PT5B_full_weightNorm_TIM.pkl',
                                      threshold=cfg.weightNormThreshold)

    # # Test that mutant is being loaded! and test Ih
    # for secName in cellRule['secs']:
    #     print(secName, cellRule['secs'][secName]['mechs']['Ih']['gIhbar'])
    #     # print(cellRule['secs'][secName]['mechs']['na12'])
    #     # print(cellRule['secs'][secName]['mechs']['na12mut'])
    # quit()

    # save to json with all the above modifications so easier/faster to load
    if saveCellParams: netParamsAux.saveCellParamsRule(label='PT5B_full', fileName='./cells/Na12HH16HH_TF.json')

    # netParamsAux.saveCellParamsRule(label='PT5B_full', fileName='./wscale_PT5B/Na12HH16HH_TF.json')


    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def PT5BFullModel(cfg, cwd, saveCellParams):
    netParamsAux = specs.NetParams()
    ihMod2str = {'harnett': 1, 'kole': 2, 'migliore': 3}
    cellRule = netParamsAux.importCellParams(label='PT5B_full', conds={'cellType': 'PT', 'cellModel': 'HH_full'},
      fileName=cwd+'/cells/PTcell.hoc', cellName='PTcell', cellArgs=[ihMod2str[cfg.ihModel], cfg.ihSlope], somaAtOrigin=True)
    nonSpiny = ['apic_0', 'apic_1']
    netParamsAux.addCellParamsSecList(label='PT5B_full', secListName='perisom', somaDist=[0, 50])  # sections within 50 um of soma
    netParamsAux.addCellParamsSecList(label='PT5B_full', secListName='below_soma', somaDistY=[-600, 0])  # sections within 0-300 um of soma
    for sec in nonSpiny: cellRule['secLists']['perisom'].remove(sec)
    cellRule['secLists']['alldend'] = [sec for sec in cellRule.secs if ('dend' in sec or 'apic' in sec)] # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule.secs if ('apic' in sec)] # apical
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in nonSpiny]
    # Adapt ih params based on cfg param
    for secName in cellRule['secs']:
        for mechName,mech in cellRule['secs'][secName]['mechs'].items():
            if mechName in ['ih','h','h15', 'hd']: 
                mech['gbar'] = [g*cfg.ihGbar for g in mech['gbar']] if isinstance(mech['gbar'],list) else mech['gbar']*cfg.ihGbar
                if cfg.ihModel == 'migliore':   
                    mech['clk'] = cfg.ihlkc  # migliore's shunt current factor
                    mech['elk'] = cfg.ihlke  # migliore's shunt current reversal potential
                if secName.startswith('dend'): 
                    mech['gbar'] *= cfg.ihGbarBasal  # modify ih conductance in soma+basal dendrites
                    mech['clk'] *= cfg.ihlkcBasal  # modify ih conductance in soma+basal dendrites
                if secName in cellRule['secLists']['below_soma']: #secName.startswith('dend'): 
                    mech['clk'] *= cfg.ihlkcBelowSoma  # modify ih conductance in soma+basal dendrites
    # Reduce dend Na to avoid dend spikes (compensate properties by modifying axon params)
    for secName in cellRule['secLists']['alldend']:
        cellRule['secs'][secName]['mechs']['nax']['gbar'] = 0.0153130368342 * cfg.dendNa # 0.25 
    cellRule['secs']['soma']['mechs']['nax']['gbar'] = 0.0153130368342  * cfg.somaNa
    cellRule['secs']['axon']['mechs']['nax']['gbar'] = 0.0153130368342  * cfg.axonNa # 11  
    cellRule['secs']['axon']['geom']['Ra'] = 137.494564931 * cfg.axonRa # 0.005
    # Remove Na (TTX)
    if cfg.removeNa:
        for secName in cellRule['secs']: cellRule['secs'][secName]['mechs']['nax']['gbar'] = 0.0
    netParamsAux.addCellParamsWeightNorm('PT5B_full', cwd+'/conn/PT5B_full_weightNorm.pkl', threshold=cfg.weightNormThreshold)  # load weight norm
    if saveCellParams: netParamsAux.saveCellParamsRule(label='PT5B_full', fileName=cwd+'/cells/PT5B_full_cellParams.pkl')

    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def IT5AFullModel(cwd, saveCellParams, cfg, layer):
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label='IT5A_full', conds={'cellType': 'IT', 'cellModel': 'HH_full', 'ynorm': layer['5A']},
    fileName=cwd+'/cells/ITcell.py', cellName='ITcell', cellArgs={'params': 'BS1579'}, somaAtOrigin=True)
    netParamsAux.renameCellParamsSec(label='IT5A_full', oldSec='soma_0', newSec='soma')
    netParamsAux.addCellParamsWeightNorm('IT5A_full', cwd+'/conn/IT_full_BS1579_weightNorm.pkl', threshold=cfg.weightNormThreshold) # add weightNorm before renaming soma_0
    netParamsAux.addCellParamsSecList(label='IT5A_full', secListName='perisom', somaDist=[0, 50])  # sections within 50 um of soma
    cellRule['secLists']['alldend'] = [sec for sec in cellRule.secs if ('dend' in sec or 'apic' in sec)] # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule.secs if ('apic' in sec)] # basal+apical
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in ['apic_0', 'apic_1']]
    if saveCellParams: netParamsAux.saveCellParamsRule(label='IT5A_full', fileName=cwd+'/cells/IT5A_full_cellParams.pkl')
    
    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def IT5BFullModel(cwd, layer, saveCellParams): # NOT USED
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label='IT5B_full', conds={'cellType': 'IT', 'cellModel': 'HH_full', 'ynorm': layer['5B']},
    fileName=cwd+'/cells/ITcell.py', cellName='ITcell', cellArgs={'params': 'BS1579'}, somaAtOrigin=True)
    netParamsAux.addCellParamsSecList(label='IT5B_full', secListName='perisom', somaDist=[0, 50])  # sections within 50 um of soma
    cellRule['secLists']['alldend'] = [sec for sec in cellRule.secs if ('dend' in sec or 'apic' in sec)] # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule.secs if ('apic' in sec)] # basal+apical
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in ['apic_0', 'apic_1']]
    netParamsAux.addCellParamsWeightNorm('IT5B_full', cwd+'/conn/IT_full_BS1579_weightNorm.pkl')
    if saveCellParams: netParamsAux.saveCellParamsRule(label='IT5B_full', fileName=cwd+'/cells/IT5B_full_cellParams.pkl')

    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def PVReducedModel(cwd, cfg, saveCellParams):
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label='PV_reduced', conds={'cellType':'PV', 'cellModel':'HH_reduced'}, 
    fileName=cwd+'/cells/FS3.hoc', cellName='FScell1', cellInstance = True)
    cellRule['secLists']['spiny'] = ['soma', 'dend']
    netParamsAux.addCellParamsWeightNorm('PV_reduced', cwd+'/conn/PV_reduced_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    # cellRule['secs']['soma']['weightNorm'][0] *= 1.5
    if saveCellParams: netParamsAux.saveCellParamsRule(label='PV_reduced', fileName=cwd+'/cells/PV_reduced_cellParams.pkl')
    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def SOMReducedModel(cwd, cfg, saveCellParams):
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label='SOM_reduced', conds={'cellType':'SOM', 'cellModel':'HH_reduced'}, 
    fileName=cwd+'/cells/LTS3.hoc', cellName='LTScell1', cellInstance = True)
    cellRule['secLists']['spiny'] = ['soma', 'dend']
    netParamsAux.addCellParamsWeightNorm('SOM_reduced', cwd+'/conn/SOM_reduced_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    if saveCellParams: netParamsAux.saveCellParamsRule(label='SOM_reduced', fileName=cwd+'/cells/SOM_reduced_cellParams.pkl')
    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def VIPReducedModel(cwd, cfg, saveCellParams):
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label='VIP_reduced', conds={'cellType': 'VIP', 'cellModel': 'HH_reduced'},
                                            fileName=cwd+'/cells/vipcr_cell.hoc',         
                                            cellName='VIPCRCell_EDITED', importSynMechs = True)
    cellRule['secLists']['spiny'] = ['soma', 'rad1', 'rad2', 'ori1', 'ori2']
    netParamsAux.addCellParamsWeightNorm('VIP_reduced', cwd+'/conn/VIP_reduced_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    if saveCellParams: netParamsAux.saveCellParamsRule(label='VIP_reduced', fileName=cwd+'/cells/VIP_reduced_cellParams.pkl')
    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def NGFReducedModel(cwd, cfg, saveCellParams):
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label='NGF_reduced', conds={'cellType': 'NGF', 'cellModel': 'HH_reduced'}, 
                                        fileName=cwd+'/cells/ngf_cell.hoc',
                                        cellName='ngfcell', importSynMechs = True)
    cellRule['secLists']['spiny'] = ['soma', 'dend']
    netParamsAux.addCellParamsWeightNorm('NGF_reduced', cwd+'/conn/NGF_reduced_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    cellRule['secs']['soma']['weightNorm'][0] *= 1.5
    cellRule['secs']['soma']['weightNorm'][0] *= 1.5
    if saveCellParams: netParamsAux.saveCellParamsRule(label='NGF_reduced', fileName=cwd+'/cells/NGF_reduced_cellParams.pkl')
    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def definePops(netParams, cfg, layer, density):
    ## Local populations
    ### Layer 1:
    netParams.popParams['NGF1']  =   {'cellModel': 'HH_reduced', 'cellType': 'NGF', 'ynormRange': layer['1'], 'density': density[('M1','nonVIP')][0]}

    ### Layer 2/3:
    netParams.popParams['IT2']  =   {'cellModel': cfg.cellmod['IT2'],  'cellType': 'IT', 'ynormRange': layer['2'], 'density': density[('M1','E')][1]}
    netParams.popParams['SOM2'] =   {'cellModel': 'HH_reduced',         'cellType': 'SOM','ynormRange': layer['2'], 'density': density[('M1','SOM')][1]}
    netParams.popParams['PV2']  =   {'cellModel': 'HH_reduced',         'cellType': 'PV', 'ynormRange': layer['2'], 'density': density[('M1','PV')][1]}
    netParams.popParams['VIP2']  =  {'cellModel': 'HH_reduced',        'cellType': 'VIP', 'ynormRange': layer['2'], 'density': density[('M1','VIP')][1]}
    netParams.popParams['NGF2']  =  {'cellModel': 'HH_reduced',         'cellType': 'NGF', 'ynormRange': layer['2'], 'density': density[('M1','nonVIP')][1]}

    ### Layer 4:
    netParams.popParams['IT4']  =   {'cellModel': cfg.cellmod['IT4'],  'cellType': 'IT', 'ynormRange': layer['4'], 'density': density[('M1','E')][2]}
    netParams.popParams['SOM4'] =   {'cellModel': 'HH_reduced',         'cellType': 'SOM','ynormRange': layer['4'], 'density': density[('M1','SOM')][2]}
    netParams.popParams['PV4']  =   {'cellModel': 'HH_reduced',         'cellType': 'PV', 'ynormRange': layer['4'], 'density': density[('M1','PV')][2]}
    netParams.popParams['VIP4']  =  {'cellModel': 'HH_reduced',        'cellType': 'VIP', 'ynormRange': layer['4'], 'density': density[('M1','VIP')][2]}
    netParams.popParams['NGF4']  =  {'cellModel': 'HH_reduced',         'cellType': 'NGF', 'ynormRange': layer['4'], 'density': density[('M1','nonVIP')][2]}

    ### Layer 5A:
    netParams.popParams['IT5A'] =   {'cellModel': cfg.cellmod['IT5A'], 'cellType': 'IT', 'ynormRange': layer['5A'], 'density': density[('M1','E')][3]}
    netParams.popParams['SOM5A'] =  {'cellModel': 'HH_reduced',         'cellType': 'SOM','ynormRange': layer['5A'], 'density': density[('M1','SOM')][3]}
    netParams.popParams['PV5A']  =  {'cellModel': 'HH_reduced',         'cellType': 'PV', 'ynormRange': layer['5A'], 'density': density[('M1','PV')][3]}
    netParams.popParams['VIP5A']  = {'cellModel': 'HH_reduced',         'cellType': 'VIP', 'ynormRange': layer['5A'], 'density': density[('M1','VIP')][3]}
    netParams.popParams['NGF5A']  = {'cellModel': 'HH_reduced',         'cellType': 'NGF', 'ynormRange': layer['5A'], 'density': density[('M1','nonVIP')][3]}

    ### Layer 5B:
    netParams.popParams['IT5B'] =   {'cellModel': cfg.cellmod['IT5B'], 'cellType': 'IT', 'ynormRange': layer['5B'], 'density': 0.5*density[('M1','E')][4]}
    netParams.popParams['PT5B'] =   {'cellModel': cfg.cellmod['PT5B'], 'cellType': 'PT', 'ynormRange': layer['5B'], 'density': 0.5*density[('M1','E')][4]}
    netParams.popParams['SOM5B'] =  {'cellModel': 'HH_reduced',         'cellType': 'SOM','ynormRange': layer['5B'], 'density': density[('M1','SOM')][4]}
    netParams.popParams['PV5B']  =  {'cellModel': 'HH_reduced',         'cellType': 'PV', 'ynormRange': layer['5B'], 'density': density[('M1','PV')][4]}
    netParams.popParams['VIP5B']  = {'cellModel': 'HH_reduced',        'cellType': 'VIP', 'ynormRange': layer['5B'], 'density': density[('M1','VIP')][4]}
    netParams.popParams['NGF5B']  = {'cellModel': 'HH_reduced',         'cellType': 'NGF', 'ynormRange': layer['5B'], 'density': density[('M1','nonVIP')][4]}

    ### Layer 6:
    netParams.popParams['IT6']  =   {'cellModel': cfg.cellmod['IT6'],  'cellType': 'IT', 'ynormRange': layer['6'],  'density': 0.5*density[('M1','E')][5]}
    netParams.popParams['CT6']  =   {'cellModel': cfg.cellmod['CT6'],  'cellType': 'CT', 'ynormRange': layer['6'],  'density': 0.5*density[('M1','E')][5]}
    netParams.popParams['SOM6'] =   {'cellModel': 'HH_reduced',         'cellType': 'SOM','ynormRange': layer['6'],  'density': density[('M1','SOM')][5]}
    netParams.popParams['PV6']  =   {'cellModel': 'HH_reduced',         'cellType': 'PV', 'ynormRange': layer['6'],  'density': density[('M1','PV')][5]}
    netParams.popParams['VIP6']  =  {'cellModel': 'HH_reduced',        'cellType': 'VIP', 'ynormRange': layer['6'], 'density': density[('M1','VIP')][1]}
    netParams.popParams['NGF6']  =  {'cellModel': 'HH_reduced',         'cellType': 'NGF', 'ynormRange': layer['6'], 'density': density[('M1','nonVIP')][1]}

    return None

def addLongConnections(cwd, netParams, cfg, layer):
    #TODO: Check and rewrite to load in-vivo spikes
    import pickle, json
    ## load experimentally based parameters for long range inputs
    with open(cwd + '/conn/conn_long.pkl', 'rb') as fileObj:
        connLongData = pickle.load(fileObj)
    # ratesLong = connLongData['rates']

    numCells = cfg.numCellsLong
    noise = cfg.noiseLong
    start = cfg.startLong

    if cfg.addInVivoThalamus: 
        longPops = ['TPO', 'S1', 'S2', 'cM1', 'M2', 'OC']
    else:
        longPops = ['TPO', 'TVL', 'S1', 'S2', 'cM1', 'M2', 'OC']
    ## create populations with fixed
    for longPop in longPops:
        netParams.popParams[longPop] = {'cellModel': 'VecStim', 'numCells': numCells, 'rate': cfg.ratesLong[longPop],
                                        'noise': noise, 'start': start, 'pulses': [],
                                        'ynormRange': layer['long' + longPop]}
        if isinstance(cfg.ratesLong[longPop], str):  # filename to load spikes from
            spikesFile = cfg.ratesLong[longPop]
            with open(spikesFile, 'r') as f: spks = json.load(f)
            netParams.popParams[longPop].pop('rate')
            netParams.popParams[longPop]['spkTimes'] = spks

    if cfg.addInVivoThalamus:   
        netParams.popParams['TVL'] = {'cellModel': 'VecStim',
                                                 'numCells': len(cfg.spikeTimesInVivo),
                                                 'spkTimes': cfg.spikeTimesInVivo,
                                                 'ynormRange': layer['long' + 'TVL']}
    return connLongData

def addStimPulses(cfg, netParams):
    for key in [k for k in dir(cfg) if k.startswith('pulse')]:
        params = getattr(cfg, key, None)
        [pop, start, end, rate, noise] = [params[s] for s in ['pop', 'start', 'end', 'rate', 'noise']]
        if 'duration' in params and params['duration'] is not None and params['duration'] > 0:
            end = start + params['duration']

        if pop in netParams.popParams:
            if 'pulses' not in netParams.popParams[pop]: netParams.popParams[pop]['pulses'] = {}    
            netParams.popParams[pop]['pulses'].append({'start': start, 'end': end, 'rate': rate, 'noise': noise})
    
    return None

def addStimIClamp(cfg, netParams):
    for key in [k for k in dir(cfg) if k.startswith('IClamp')]:
        params = getattr(cfg, key, None)
        [pop,sec,loc,start,dur,amp] = [params[s] for s in ['pop','sec','loc','start','dur','amp']]

        #cfg.analysis['plotTraces']['include'].append((pop,0))  # record that pop

        # add stim source
        netParams.stimSourceParams[key] = {'type': 'IClamp', 'delay': start, 'dur': dur, 'amp': amp}
        
        # connect stim source to target
        netParams.stimTargetParams[key+'_'+pop] =  {
            'source': key, 
            'conds': {'pop': pop},
            'sec': sec, 
            'loc': loc}
    return None

def addStimNetStim(cfg, netParams, ESynMech, SOMESynMech):
    for key in [k for k in dir(cfg) if k.startswith('NetStim')]:
        params = getattr(cfg, key, None)
        [pop, ynorm, sec, loc, synMech, synMechWeightFactor, start, interval, noise, number, weight, delay] = \
        [params[s] for s in ['pop', 'ynorm', 'sec', 'loc', 'synMech', 'synMechWeightFactor', 'start', 'interval', 'noise', 'number', 'weight', 'delay']] 

        # cfg.analysis['plotTraces']['include'] = [(pop,0)]

        if synMech == ESynMech:
            wfrac = cfg.synWeightFractionEE
        elif synMech == SOMESynMech:
            wfrac = cfg.synWeightFractionSOME
        else:
            wfrac = [1.0] #TODO: What is the use of wfrac??

        # add stim source
        netParams.stimSourceParams[key] = {'type': 'NetStim', 'start': start, 'interval': interval, 'noise': noise, 'number': number}

        # connect stim source to target
        # for i, syn in enumerate(synMech):
        netParams.stimTargetParams[key+'_'+pop] =  {
            'source': key, 
            'conds': {'pop': pop, 'ynorm': ynorm},
            'sec': sec, 
            'loc': loc,
            'synMech': synMech,
            'weight': weight,
            'synMechWeightFactor': synMechWeightFactor,
            'delay': delay}
    return None

def defineEEConnections(bins, cfg, netParams, cellModels, pmat, wmat):
    labelsConns = [('W+AS_norm', 'IT', 'L2/3,4'), ('W+AS_norm', 'IT', 'L5A,5B'), 
                   ('W+AS_norm', 'PT', 'L5B'), ('W+AS_norm', 'IT', 'L6'), ('W+AS_norm', 'CT', 'L6')]
    labelPostBins = [('W+AS', 'IT', 'L2/3,4'), ('W+AS', 'IT', 'L5A,5B'), ('W+AS', 'PT', 'L5B'), 
                    ('W+AS', 'IT', 'L6'), ('W+AS', 'CT', 'L6')]
    labelPreBins = ['W', 'AS', 'AS', 'W', 'W']
    preTypes = [['IT'], ['IT'], ['IT', 'PT'], ['IT','CT'], ['IT','CT']] 
    postTypes = ['IT', 'IT', 'PT', 'IT','CT']
    ESynMech = ['AMPA','NMDA']

    for i,(label, preBinLabel, postBinLabel) in enumerate(zip(labelsConns,labelPreBins, labelPostBins)):
        for ipre, preBin in enumerate(bins[preBinLabel]):
            for ipost, postBin in enumerate(bins[postBinLabel]):
                for cellModel in cellModels:
                    ruleLabel = 'EE_'+cellModel+'_'+str(i)+'_'+str(ipre)+'_'+str(ipost)
                    netParams.connParams[ruleLabel] = { 
                        'preConds': {'cellType': preTypes[i], 'ynorm': list(preBin)}, 
                        'postConds': {'cellModel': cellModel, 'cellType': postTypes[i], 'ynorm': list(postBin)},
                        'synMech': ESynMech,
                        'probability': pmat[label][ipost,ipre],
                        'weight': wmat[label][ipost,ipre] * cfg.EEGain / cfg.synsperconn[cellModel], 
                        'synMechWeightFactor': cfg.synWeightFractionEE,
                        'delay': 'defaultDelay+dist_3D/propVelocity',
                        'synsPerConn': cfg.synsperconn[cellModel],
                        'sec': 'spiny'}

    return None

def defineEIConnections(excTypes, inhTypes, bins, cfg, netParams, pmat, wmat):
    binsLabel = 'inh'
    preTypes = excTypes
    postTypes = inhTypes
    ESynMech = ['AMPA','NMDA']
    for i,postType in enumerate(postTypes):
        for ipre, preBin in enumerate(bins[binsLabel]):
            for ipost, postBin in enumerate(bins[binsLabel]):
                ruleLabel = 'EI_'+str(i)+'_'+str(ipre)+'_'+str(ipost)+'_'+str(postType)
                netParams.connParams[ruleLabel] = {
                    'preConds': {'cellType': preTypes, 'ynorm': list(preBin)},
                    'postConds': {'cellType': postType, 'ynorm': list(postBin)},
                    'synMech': ESynMech,
                    'probability': pmat[('E', postType)][ipost,ipre],
                    'weight': wmat[('E', postType)][ipost,ipre] * cfg.EIGain * cfg.EICellTypeGain[postType],
                    'synMechWeightFactor': cfg.synWeightFractionEI,
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'sec': 'soma'} # simple I cells used right now only have soma
    return None

def defineIEConnections(excTypes, inhTypes, bins, cfg, netParams, pmat, PVSynMech, SOMESynMech, VIPSynMech, NGFSynMech):
    binsLabel = 'inh'
    preTypes = inhTypes
    synMechs = [PVSynMech, SOMESynMech, VIPSynMech, NGFSynMech] 
    weightFactors = [[1.0], cfg.synWeightFractionSOME, [1.0], cfg.synWeightFractionNGF] # Update VIP and NGF syns! 
    secs = ['perisom', 'apicdend', 'apicdend', 'apicdend']
    postTypes = excTypes
    for ipreType, (preType, synMech, weightFactor, sec) in enumerate(zip(preTypes, synMechs, weightFactors, secs)):
        for ipostType, postType in enumerate(postTypes):
            for ipreBin, preBin in enumerate(bins[binsLabel]):
                for ipostBin, postBin in enumerate(bins[binsLabel]):
                    for cellModel in ['HH_reduced', 'HH_full']:
                        ruleLabel = preType+'_'+postType+'_'+cellModel+'_'+str(ipreBin)+'_'+str(ipostBin)
                        netParams.connParams[ruleLabel] = {
                            'preConds': {'cellType': preType, 'ynorm': list(preBin)},
                            'postConds': {'cellModel': cellModel, 'cellType': postType, 'ynorm': list(postBin)},
                            'synMech': synMech,
                            'probability': '%f * exp(-dist_3D_border/probLambda)' % (pmat[(preType, 'E')][ipostBin,ipreBin]),
                            'weight': cfg.IEweights[ipostBin] * cfg.IEGain/ cfg.synsperconn[cellModel],
                            'synMechWeightFactor': weightFactor,
                            'synsPerConn': cfg.synsperconn[cellModel],
                            'delay': 'defaultDelay+dist_3D/propVelocity',
                            'sec': sec} # simple I cells used right now only have soma
    return None

def defineIIConnections(excTypes, inhTypes, bins, cfg, netParams, pmat, PVSynMech, SOMESynMech, VIPSynMech, NGFSynMech):
    binsLabel = 'inh'
    preTypes = inhTypes
    synMechs =  [PVSynMech, SOMESynMech, VIPSynMech, NGFSynMech]   
    sec = 'perisom'
    postTypes = inhTypes
    for ipre, (preType, synMech) in enumerate(zip(preTypes, synMechs)):
        for ipost, postType in enumerate(postTypes):
            for iBin, bin in enumerate(bins[binsLabel]):
                for cellModel in ['HH_reduced']:
                    ruleLabel = preType+'_'+postType+'_'+str(iBin)
                    netParams.connParams[ruleLabel] = {
                        'preConds': {'cellType': preType, 'ynorm': bin},
                        'postConds': {'cellModel': cellModel, 'cellType': postType, 'ynorm': bin},
                        'synMech': synMech,
                        'probability': '%f * exp(-dist_3D_border/probLambda)' % (pmat[(preType, postType)]),
                        'weight': cfg.IIweights[iBin] * cfg.IIGain / cfg.synsperconn[cellModel],
                        'synsPerConn': cfg.synsperconn[cellModel],
                        'delay': 'defaultDelay+dist_3D/propVelocity',
                        'sec': sec} # simple I cells used right now only have soma
    return None

def defineLongRangeConnections(connLongData, cfg, netParams, cellModels, ESynMech):
    # load load experimentally based parameters for long range inputs
    cmatLong = connLongData['cmat']
    binsLong = connLongData['bins']

    longPops = ['TPO', 'TVL', 'S1', 'S2', 'cM1', 'M2', 'OC']
    cellTypes = ['IT', 'PT', 'CT', 'PV', 'SOM', 'VIP', 'NGF']
    EorI = ['exc', 'inh']
    syns = {'exc': ESynMech, 'inh': 'GABAA'}
    synFracs = {'exc': cfg.synWeightFractionEE, 'inh': [1.0]}

    for longPop in longPops:
        for ct in cellTypes:
            for EorI in ['exc', 'inh']:
                for i, (binRange, convergence) in enumerate(zip(binsLong[(longPop, ct)], cmatLong[(longPop, ct, EorI)])):
                    for cellModel in cellModels:
                        ruleLabel = longPop+'_'+ct+'_'+EorI+'_'+cellModel+'_'+str(i)
                        netParams.connParams[ruleLabel] = { 
                            'preConds': {'pop': longPop}, 
                            'postConds': {'cellModel': cellModel, 'cellType': ct, 'ynorm': list(binRange)},
                            'synMech': syns[EorI],
                            'convergence': convergence,
                            'weight': cfg.weightLong[longPop] / cfg.synsperconn[cellModel], 
                            'synMechWeightFactor': cfg.synWeightFractionEE,
                            'delay': 'defaultDelay+dist_3D/propVelocity',
                            'synsPerConn': cfg.synsperconn[cellModel],
                            'sec': 'spiny'}
    return None

def defineSubcellularConnectivity(cwd, netParams, layer, ESynMech, SOMESynMech, VIPSynMech, NGFSynMech, inhTypes):
    import json
    with open(cwd+'/conn/conn_dend_PT.json', 'r') as fileObj: connDendPTData = json.load(fileObj)
    with open(cwd+'/conn/conn_dend_IT.json', 'r') as fileObj: connDendITData = json.load(fileObj)
    
    #------------------------------------------------------------------------------
    # L2/3,TVL,S2,cM1,M2 -> PT (Suter, 2015)
    lenY = 30 
    spacing = 50
    gridY = list(range(0, -spacing*lenY, -spacing))
    synDens, _, fixedSomaY = connDendPTData['synDens'], connDendPTData['gridY'], connDendPTData['fixedSomaY']
    for k in synDens.keys():
        prePop,postType = k.split('_')  # eg. split 'M2_PT'
        if prePop == 'L2': prePop = 'IT2'  # include conns from layer 2/3 and 4
        netParams.subConnParams[k] = {
        'preConds': {'pop': prePop}, 
        'postConds': {'cellType': postType},  
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': {'type': '1Dmap', 'gridX': None, 'gridY': gridY, 'gridValues': synDens[k], 'fixedSomaY': fixedSomaY}} 

    #------------------------------------------------------------------------------
    # TPO, TVL, M2, OC  -> E (L2/3, L5A, L5B, L6) (Hooks 2013)
    lenY = 26
    spacing = 50
    gridY = list(range(0, -spacing*lenY, -spacing))
    synDens, _, fixedSomaY = connDendITData['synDens'], connDendITData['gridY'], connDendITData['fixedSomaY']
    for k in synDens.keys():
        prePop,post = k.split('_')  # eg. split 'M2_L2'
        postCellTypes = ['IT','PT','CT'] if prePop in ['OC','TPO'] else ['IT','CT']  # only OC,TPO include PT cells
        postyRange = list(layer[post.split('L')[1]]) # get layer yfrac range 
        if post == 'L2': postyRange[1] = layer['4'][1]  # apply L2 rule also to L4 
        netParams.subConnParams[k] = {
        'preConds': {'pop': prePop}, 
        'postConds': {'ynorm': postyRange , 'cellType': postCellTypes},  
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': {'type': '1Dmap', 'gridX': None, 'gridY': gridY, 'gridValues': synDens[k], 'fixedSomaY': fixedSomaY}} 

    #------------------------------------------------------------------------------
    # S1, S2, cM1 -> E IT/CT; no data, assume uniform over spiny
    netParams.subConnParams['S1,S2,cM1->IT,CT'] = {
        'preConds': {'pop': ['S1','S2','cM1']}, 
        'postConds': {'cellType': ['IT','CT']},
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # rest of local E->E (exclude IT2->PT); uniform distribution over spiny
    netParams.subConnParams['IT2->non-PT'] = {
        'preConds': {'pop': ['IT2']}, 
        'postConds': {'cellType': ['IT','CT']},
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'} 
        
    netParams.subConnParams['non-IT2->E'] = {
        'preConds': {'pop': ['IT4','IT5A','IT5B','PT5B','IT6','CT6']}, 
        'postConds': {'cellType': ['IT','PT','CT']},
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # PV->E; perisomatic (no sCRACM)
    netParams.subConnParams['PV->E'] = {
        'preConds': {'cellType': 'PV'}, 
        'postConds': {'cellType': ['IT', 'CT', 'PT']},  
        'sec': 'perisom', 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # SOM->E; apical dendrites (no sCRACM)
    netParams.subConnParams['SOM->E'] = {
        'preConds': {'cellType': 'SOM'}, 
        'postConds': {'cellType': ['IT', 'CT', 'PT']},  
        'sec': 'apicdend',
        'groupSynMechs': SOMESynMech,
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # VIP->E; apical dendrites (no sCRACM)
    netParams.subConnParams['VIP->E'] = {
        'preConds': {'cellType': 'VIP'}, 
        'postConds': {'cellType': ['IT', 'CT', 'PT']},  
        'sec': 'apicdend',
        'groupSynMechs': VIPSynMech,
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # NGF->E; apical dendrites (no sCRACM)
    ## Add the following level of detail?
    # -- L1 NGF -> L2/3+L5 tuft
    # -- L2/3 NGF -> L2/3+L5 distal apical
    # -- L5 NGF -> L5 prox apical
    netParams.subConnParams['NGF->E'] = {
        'preConds': {'cellType': 'NGF'}, 
        'postConds': {'cellType': ['IT', 'CT', 'PT']},  
        'sec': 'apicdend',
        'groupSynMechs': NGFSynMech,
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # All->I; apical dendrites (no sCRACM)
    netParams.subConnParams['All->I'] = {
        'preConds': {'cellType': ['IT', 'CT', 'PT'] + inhTypes},# + longPops}, 
        'postConds': {'cellType': inhTypes},  
        'sec': 'spiny',
        'groupSynMechs': ESynMech,
        'density': 'uniform'} 

    return None

def SampleSpikes(spikeTimesList, cfg, preTone=-2., postTone=2, baselineEnd=-0.5, skipEmpty=False):
    # Check that the spiking input is enough to run the simulation
    if (cfg.SimulateBaseline==False and cfg.preTone>2000.):
        raise ValueError("cfg.preTone cannot be larger than 2000 ms") # TODO: Add extension for preTone: we could add more baseline to the left actually
    if (cfg.SimulateBaseline==False and cfg.postTone>2000.):
        raise ValueError("cfg.preTone cannot be larger than 2000 ms") # TODO: Add extension for postTone: could it be baseline again?

    MovementTrials = []
    BaselineTrials = []
    for spkList in spikeTimesList:
        MovementTrialsAux = []
        BaselineTrialsAux = []
        for spkTimes in spkList:
            if (preTone <= spkTimes <= baselineEnd): BaselineTrialsAux.append(1000*(spkTimes+abs(preTone)))
            if (-cfg.preTone/1000. <= spkTimes <= cfg.postTone/1000.): 
                PositiveTimes = 1000*spkTimes+cfg.preTone
                MovementTrialsAux.append(PositiveTimes)
        if skipEmpty:
            if len(MovementTrialsAux)>0: MovementTrials.append(MovementTrialsAux)
            if len(BaselineTrialsAux)>0: BaselineTrials.append(BaselineTrialsAux)
        else:
            MovementTrials.append(MovementTrialsAux)
            BaselineTrials.append(BaselineTrialsAux)
    # Sample spikes
    random.seed(cfg.seeds['tvl_sampling'])
    baselineSpks = random.choices(BaselineTrials, k=cfg.numCellsLong)
    baselineSpks = [list(i) for i in baselineSpks]

    movementAndPostSpks = random.choices(MovementTrials, k=cfg.numCellsLong)
    movementAndPostSpks = [list(i) for i in movementAndPostSpks]

    if cfg.SimulateBaseline==True:
        sampledSpikesSpan = 1000*(baselineEnd-preTone)
        numSpans = math.ceil(cfg.duration / sampledSpikesSpan)
        # TODO: Check the spike times extension (times should be unique and ordered)
        for num in range(numSpans-1):
            # Add a new sampling
            baselineSpksAux = random.choices(BaselineTrials, k=cfg.numCellsLong)
            baselineSpksAux = [[elem + (num+1)*sampledSpikesSpan for elem in sublist] for sublist in baselineSpksAux]
            baselineSpks = [a + b for a, b in zip(baselineSpks, baselineSpksAux)]

    return baselineSpks, movementAndPostSpks

def cellPerlayer(numbers):
    Layers = {'1': [0.0, 0.1*1350], '2': [0.1*1350,0.29*1350], '4': [0.29*1350,0.37*1350], '5A': [0.3*1350,0.47*1350], '5B': [0.47*1350,0.8*1350], '6': [0.8*1350, 1.0*1350]}

    from collections import defaultdict

    counts = defaultdict(int)

    for num in numbers:
        for layer, (low, high) in Layers.items():
            if low <= num < high:
                counts[layer] += 1
                break  # Assumes one number belongs to only one layer

    return counts

def loadThalSpikes(cwd, cfg, skipEmpty=False):
    import json
    with open(cwd+"/data/spikingData/ThRates.json", "r") as fileObj:
        data = json.loads(fileObj.read())

    spikeTimesList = []
    M1sampledCells = []
    foldersName = []

    for folder in data.keys():
        for i in range(len(data[folder].keys())-4): # exclude M1_cell_depths, Th_cell_depths, meanRate, stdRate
            spkid =  data[folder]['trial_%d' % i]['spkid']
            spkt = data[folder]['trial_%d' % i]['spkt']
            npre = int(np.max(spkid)) + 1
            spkTimes_by_cell = [[] for _ in range(npre)]
            for t, i in zip(spkt, spkid):
                spkTimes_by_cell[int(i)].append(float(t))
            spikeTimesList[len(spikeTimesList):] += spkTimes_by_cell
        cellDepths = data[folder]['M1_cell_depths']
        counts = cellPerlayer(cellDepths)
        M1sampledCells.append(counts)
        foldersName.append(folder)

    baselineSpks, movementAndPostSpks = SampleSpikes(spikeTimesList, cfg, skipEmpty=skipEmpty)

    return baselineSpks, movementAndPostSpks, M1sampledCells, foldersName

def average_dict_entries(dicts: List[Union[dict, defaultdict]]) -> Dict[str, float]:
    totals = defaultdict(int)
    counts = defaultdict(int)

    for entry in dicts:
        for key, value in entry.items():
            totals[key] += value
            counts[key] += 1

    averages = {key: int(totals[key] / counts[key]) for key in totals}
    return averages

def trimTVLSpikes(spikeList, cfg):
    trimmedList = []
    for i in spikeList:
        # We need to align the spike time to avoid numerical errors in the delivery of the vecStim (due torounding errors it could happen that the simulator find a negative delivery time, which stops the simulation)
        trimmedList.append(np.unique([round(np.round(j / cfg.dt) * cfg.dt, 2) for j in i if (0<j<cfg.duration)]).tolist())

    return trimmedList

