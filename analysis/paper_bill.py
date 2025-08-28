"""
paper.py 

Paper figures

Contributors: salvadordura@gmail.com
"""

import utils
import json
import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb
import os
import pickle
import batchAnalysis as ba
from netpyne.support.scalebar import add_scalebar

import IPython

#plt.ion()  # interactive

# ---------------------------------------------------------------------------------------------------------------
# Population params
allpops = ['IT2','SOM2','PV2','IT4','IT5A','SOM5A','PV5A','IT5B','PT5B','SOM5B','PV5B',
    'IT6','CT6','SOM6','PV6']
excpops = ['IT2','IT4','IT5A','IT5B','PT5B','IT6','CT6']
inhpops = ['SOM2','PV2', 'SOM5A','PV5A', 'SOM5B','PV5B',  'SOM6', 'PV6']
excpopsu = ['IT2','IT4','IT5A','PT5B']
ITsubset = ('IT2', 'IT4', 'IT5A', 'IT6')
SOMsubset = ('SOM2', 'SOM5A','SOM5B', 'SOM6')
PVsubset = ('PV2', 'PV5A', 'PV5B', 'PV6')

with open('../sim/cells/popColors.pkl', 'r') as fileObj: 
    popColors = pickle.load(fileObj)['popColors']
popColors['S2'] = [0.90,0.76,0.00]
popColors['M2'] = [0.42,0.67,0.9]
popColors[ITsubset] = popColors['IT5A']
popColors[SOMsubset] = popColors['SOM5A']
popColors[PVsubset] = popColors['PV5A']

plt.style.use('seaborn-ticks')     

def loadSimData(dataFolder, batchLabel, simLabel):
    ''' load sim file'''

    root = dataFolder+batchLabel+'/'
    sim,data,out = None, None, None
    if isinstance(simLabel, basestring): 
        filename = root+simLabel+'.json'
        print filename
        sim,data,out = utils.plotsFromFile(filename, 
            raster=0, stats=0, rates=0, syncs=0, hist=0, psd=0, traces=0, grang=0, plotAll=0)
    
    return sim, data, out, root

def axisFontSize(ax, fontsize):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 


# ---------------------------------------------------------------------------------------------------------------
def raster(timeRange):
        #[2000, 4000]
        include = allpops
        orderBy = ['pop', 'y']
        #filename = '%s%s_raster_%d_%d_%s.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
        fig1 = sim.analysis.plotRaster(include=include, timeRange=timeRange, labels='overlay', 
            popRates=0, orderInverse=True, lw=0, markerSize=3.5, marker='.', popColors=popColors, 
            showFig=0, saveFig=0, figSize=(8.5,10), orderBy=orderBy)# 
        ax = plt.gca()

        [i.set_linewidth(0.5) for i in ax.spines.itervalues()] # make border thinner
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')  #remove ticks
        plt.ylabel(' ') #Neurons (ordered by NCD within each pop)')
        plt.xlabel(' ')
        
        plt.title('')
        filename='%s%s_raster_%d_%d_%s.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
        plt.savefig(filename, dpi=600)


# Main code
if __name__ == '__main__': 

    dataFolder = '../data/'
    batchLabel = 'v53_batch11' #'v53_batch12' 
    simLabel = 'v53_batch11_0_0' #'v53_batch12_0_0_0'    
    sim, data, out, root = loadSimData(dataFolder, batchLabel, simLabel)

    timeRange = [2000, 4000] 
    raster(timeRange)
        