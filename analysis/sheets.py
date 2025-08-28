"""
sheets.py 

Reproduce Sehhets 2011 figures

Contributors: salvadordura@gmail.com
"""

import utils
import json
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb
import os
import batchAnalysis as ba
from netpyne import sim


def fig4():
    dataFolder = '../data/'
    batchLabel = 'v40_batch12'
    loadAll = 1

    pars=['ihX', 'ihY']
    vals = ['EPSPamp_PTih', 'EPSPamp_PTzd']
    query='0.00029 < groupWeight < 0.00031 and ihFactor1==2.0'
    var = [('simData','EPSPamp') , ('simData','popRates')]

    params, data = utils.readBatchData(dataFolder, batchLabel, loadAll=loadAll, saveAll=1-loadAll, vars=var, maxCombs=None) 
    
    # convert to pandas and add EPSP data
    df1 = utils.toPandas(params, data)
    if 'ratioDiff' in vals: df2 = ba.dfEPSPratios(df1)
    elif 'EPSPamp_PTih' in vals: df2 = ba.dfEPSPamps(df1)

    if query: df2 = df2.query(query)

    # plot param grids
    fig = ba.plot2Dparams(df2, par1=pars[0], par2=pars[1], val=vals, valLabel='(mV)',  groupStat='last', normRange=1, saveFile='../data/sheets/fig4c.png')
        

def fig11():       
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm

    dataFolder = '../data/v48_manualTune/'
    simLabel = 'v48_tune174'
    outFile = '../data/sheets/fig11.png'

    sim.load(dataFolder+simLabel+'.json', instantiate=False)

    # plot param grids
    fig = sim.analysis.plotTraces(include=[0,1], colors=[[0,0,0], [0,0,0]], oneFigPer= 'trace', overlay=0, figSize=(7,5), timeRange=[0,5000], ylim=[-90, 25], saveFig=False, showFig=0)

    # fig.remove axis and add scale bar
    axes = fig.values()[0].get_axes()
    titles = ['Control', '+ ZD7288']
    for i,ax in enumerate(axes):
        ax.axis('off')
        fontprops = fm.FontProperties(size=22)
        ax.set_title(titles[i], fontSize=22)
    ax = axes[1]

    scalebarHor = AnchoredSizeBar(ax.transData,
                           1000, '1 sec', loc=4, 
                           pad=-2.0,
                           color='black',
                           frameon=False,
                           size_vertical=0,
                           borderpad=0.5,
                           sep=5,
                           fontproperties=fontprops)
    ax.add_artist(scalebarHor)

    # scalebarVer = AnchoredSizeBar(ax.transData,
    #                        10, '10 mV', loc=8, 
    #                        pad=-1.5,
    #                        color='black',
    #                        frameon=False,
    #                        size_vertical=10,
    #                        borderpad=0.5,
    #                        sep=5,
    #                        fontproperties=fontprops)
    # ax.add_artist(scalebarVer)

    #plt.show()
    plt.savefig(outFile)
        
def fig2_George09():
    dataFolder = '../data/'
    batchLabel = 'v48_batch6'
    loadAll = 1

    df = ba.ihEPSPAnalysis(dataFolder, batchLabel, loadAll, pars=['groupWeight','ihGbar'], vals=['Vpeak_PTih'], zdcomp=0, plotLine=1)
    

# Main code
if __name__ == '__main__':

    #fig4()
    #fig7()
    #s=fig11()
    fig2_George09()
