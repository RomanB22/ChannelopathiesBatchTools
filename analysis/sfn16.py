"""
sfn16.py 

Code to generate analysis+figures for SFN'16 poster 

Dura-Bernal et al, 2016 "Modeling the subcellular distribution of synaptic connections in cortical microcircuits ""

Contributors: salvadordura@gmail.com
"""

import utils
import numpy as np
from itertools import product
import json

def matchConds(rates, conds = ['all']):
    ranges = {}
    # ranges['IT2'] =  [1,8]
    ranges['PV2'] =  [2.5,40]
    ranges['SOM2'] = [2.5,40]
    # ranges['IT4'] =  [0,30]
    # ranges['IT5A'] = [6,12]
    # ranges['IT5B'] = [2,12]
    # ranges['PT5B'] = [6,14]
    # ranges['IT6'] =  [4,10]
    # ranges['CT6'] =  [2,10]
    ranges['PV5'] =  [2.5,40]
    ranges['SOM5'] = [2.5,40]
    ranges['PV6'] =  [2.5,40]
    ranges['SOM6'] = [2.5,40]

    ranges['IT2'] =  [1,20]
    ranges['IT5A'] = [1,20]
    ranges['IT5B'] = [0.05,20]  
    ranges['PT5B'] = [1,30] 


    rateL5E = np.mean([rates['IT5A'], rates['IT5B'], rates['PT5B']])
    rateL6E = np.mean([rates['IT6'], rates['CT6']])
    
    # all
    if 'all' in conds:
        return True
        
    # check rate ranges
    if 'range' in conds:
        for key,rangeRate in ranges.iteritems():
            if rates[key] < rangeRate[0] or rates[key] > rangeRate[1]: 
                return False

    # check rate relation: I > E in each layer
    if 'I>E' in conds:
        if rates['IT2'] > rates['PV2'] or rates['IT2'] > rates['SOM2']:
            return False

        if rateL5E > rates['PV5'] or rateL5E > rates['SOM5']:
            return False
        
        
        if rateL6E > rates['PV6'] or rateL6E > rates['SOM6']:
            return False

    # check rate relation: I > E in each layer
    if 'I>E (L2+6)' in conds:
        if rates['IT2'] > rates['PV2'] or rates['IT2'] > rates['SOM2']:
            return False
        
        if rateL6E > rates['PV6'] or rateL6E > rates['SOM6']:
            return False

    # check rate relation: E L5 > L6 > L2
    if 'L5>L6>L2' in conds:
        if rates['IT2'] > rateL6E or rateL6E > rateL5E:
            return False

    if 'L5>L2' in conds:
        if rates['IT2'] > rateL5E:
            return False

    if 'L5>L6' in conds:
        if rateL6E > rateL5E:
            return False

    if 'L6>L2' in conds:
        if rates['IT2'] > rateL6E:
            return False

    # check rate relation: PV > LTS in each layer
    if 'PV>LTS' in conds:
        if rates['PV2'] < rates['SOM2'] or rates['PV5'] < rates['SOM5'] or rates['PV6'] < rates['SOM6']:
            return False

    return True

def netRates(params, data, plot=False, saveMatch=None):
    counter = {}
    condsList = []
    condsList.append(['all'])
    # condsList.append(['range'])
    # condsList.append(['I>E'])
    # condsList.append(['I>E (L2+6)'])
    # condsList.append(['L5>L6>L2'])
    # condsList.append(['L5>L6'])
    # condsList.append(['L5>L2'])
    # condsList.append(['L6>L2'])
    # condsList.append(['PV>LTS'])
    # condsList.append(['I>E (L2+6)', 'L5>L6>L2'])
    # condsList.append(['I>E (L2+6)', 'L5>L6>L2', 'PV>LTS'])
    # condsList.append(['I>E (L2+6)', 'L5>L2']) 
    # condsList.append(['I>E (L2+6)', 'L5>L2', 'PV>LTS'])
    # condsList.append(['range', 'I>E (L2+6)', 'L5>L6>L2', 'PV>LTS'])
    # condsList.append(['range', 'I>E (L2+6)', 'L5>L2', 'PV>LTS'])
    # condsList.append(['I>E (L2+6)', 'L5>L6>L2'])
    # condsList.append(['range', 'L5>L6>L2'])
    # condsList.append(['range', 'I>E (L2+6)', 'L5>L2'])
    # condsList.append(['range', 'I>E (L2+6)', 'L5>L2'])


    paramsMatch = []
    for iconds, conds in enumerate(condsList):
        counter[iconds] = 0
        for key, d in data.iteritems():
            # try:
            rates = d['simData']['popRates']
            if matchConds(rates, conds = conds):
                counter[iconds] = counter[iconds] + 1
                if plot:
                    print '\nMatch: %d' % (counter[iconds])
                    textTop = ''
                    for key, value in zip([x['label'] for x in params], d['paramValues']):
                        textTop = textTop + ' ' + str(key) + '=' + str(value) 
                    print textTop
                    utils.plotsFromData(d, textTop, raster=0, hist=0, psd=0, grang=1)
                #paramsMatch.append(d['paramValues'])
            # except:
            #   print 'Error in match %d' % (counter[iconds])
        print('%d / %d match firing rate conditions:' % (counter[iconds], len(data))),
        print(conds)

    if saveMatch:
        with open(saveMatch, 'w') as fileObj:
            json.dump({'paramsMatch': paramsMatch}, fileObj)


def saveStats(params, data, ratePT=True, betaPower=True, nTE=True, Granger=True):
    stats = {'uniform': {}, '1Dmap': {}, '2Dmap': {}}
    for k,v in stats.iteritems():
        for stat in ['ratePT', 'betaPower', 'nTE', 'Granger']:
            v[stat] = []

    for key, d in data.iteritems():
        rates = d['simData']['popRates']
        spkt = d['simData']['spkt']
        spkid = d['simData']['spkt']
        spksPop, numCellsPop = utils.getSpksPop(d)

        stype = d['paramValues'][0]
        print d['paramValues']

        if ratePT:
            print rates['PT5B']
            stats[stype]['ratePT'].append(rates['PT5B'])
            
        if betaPower:
            from matplotlib import mlab
            import pylab
            timeRange = [0,2000]
            binSize = 5
            histo = pylab.histogram(spksPop['PT5B'], bins = pylab.arange(timeRange[0], timeRange[1], binSize))
            histoCount = histo[0] 
            histoCount = histoCount * (1000.0 / binSize) / numCellsPop['PT5B'] # convert to rates
            power = mlab.psd(histoCount, Fs=160, NFFT=256, detrend=mlab.detrend_none, window=mlab.window_hanning, 
            noverlap=0, pad_to=None, sides='default', scale_by_freq=None)
            powerlog = 10*pylab.log10(power[0])
            maxPower = max(powerlog[13:48]) # beta power = 13-30 Hz [21:48]
            print maxPower
            stats[stype]['betaPower'].append(maxPower)

        if nTE:
            normTE = utils.nTE(spksPop['IT2'], spksPop['PT5B'])
            if normTE < 0: normTE = 0
            print normTE
            stats[stype]['nTE'].append(normTE)
    
        if Granger:
            F, Fx2y, Fy2x, Fxy = utils.granger(spksPop['IT2'], spksPop['PT5B'])
            granger = max(Fx2y[13:30])
            print granger
            stats[stype]['Granger'].append(granger)

    filename = '%s/%s/%s_stats.json' % (dataFolder, batchLabel, batchLabel)
    with open(filename, 'w') as fileObj:
        json.dump(stats, fileObj)


def plotStats(dataFolder, batchLabel, ratePT=True, betaPower=True, nTE=True, Granger=True):
    # load data
    stypes = ['uniform', '2Dmap', '1Dmap']
    print '\nLoading stat data...'
    filename = '%s/%s/%s_stats.json' % (dataFolder, batchLabel, batchLabel)
    with open(filename, 'r') as fileObj:
        dataLoad = json.load(fileObj)

    labels = ['uniform', 'map (2D)', 'radial (1D)']

    if ratePT:
        filename = '%s/%s/%s_boxplot_ratePT.png' % (dataFolder, batchLabel, batchLabel)
        utils.boxplot([dataLoad[stype]['ratePT'] for stype in stypes], labels, filename)       

    if betaPower:
        filename = '%s/%s/%s_boxplot_betaPower.png' % (dataFolder, batchLabel, batchLabel)
        utils.boxplot([dataLoad[stype]['betaPower'] for stype in stypes], labels, filename)
    
    if nTE:
        filename = '%s/%s/%s_boxplot_nTE.png' % (dataFolder, batchLabel, batchLabel)
        utils.boxplot([dataLoad[stype]['nTE'] for stype in stypes], labels, filename)

    if Granger:
        filename = '%s/%s/%s_boxplot_Granger.png' % (dataFolder, batchLabel, batchLabel)
        utils.boxplot([dataLoad[stype]['Granger'] for stype in stypes], labels, filename)
        
# main code
if __name__ == '__main__':

    dataFolder = '../data/'

    do = 'batch'

    if do == 'batch':
        # list of batch sims: label, load, save
        batchSims = []
        batchSims.append(['v20_batch6', 1, 0])
        # batchSims.append(['v20_batch7', 1, 0])
        # batchSims.append(['v20_batch7', 1, 0])
        # batchSims.append(['v20_batch8', 1, 0])
        # batchSims.append(['v20_batch9', 1, 0])
        # batchSims.append(['v20_batch10', 1, 0])

        for batchSim in batchSims:
            batchLabel = batchSim[0]
            load = batchSim[1]
            save = batchSim[2]
            filename = dataFolder+'/'+batchLabel+'/'+batchLabel+'_match.json'
            
            print '\n'+batchLabel
            #params, data = utils.readBatchData(dataFolder, batchLabel, loadAll=load, saveAll=save, vars=None, maxCombs=None) #, listCombs=filename)
            #saveStats(params, data)
            plotStats(dataFolder, batchLabel)
            #netRates(params, data, plot=1)#, saveMatch=filename)



    elif do == 'single':
        listCombs = ['1_1_1_0_1_1', '2_2_1_0_1_2', '0_0_1_0_1_2']

        batchLabel = 'v17_batch1'
        for comb in listCombs:
            filename = dataFolder+'/'+batchLabel+'/'+batchLabel+'_'+comb+'.json'
            utils.plotsFromFile(filename, raster=1, hist=0, psd=0)

    else:
        import random
        spk1=[random.randint(0,20) for i in range(10000)]
        spk2=[random.randint(0,20) for i in range(10000)]
        
        utils.nTE(spk1,spk2)
        utils.granger(spk1,spk2)