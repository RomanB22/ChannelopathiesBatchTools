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
from matplotlib import cm
from matplotlib.colors import ListedColormap

import IPython as ipy

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
try:
    with open('../sim/cells/popColors.pkl', 'rb') as fileObj: 
        popColors = pickle.load(fileObj)['popColors']
    popColors['S2'] = [0.90,0.76,0.00]
    popColors['TPO'] = [52/255.0, 138/255.0, 49/255.0] #[232/255.0, 37/255.0, 101/255.0] #'firebrick' #popColors['S2'] #[253/255.0, 102/255.0, 2/255.0] #
    popColors['M2'] = [0.42,0.67,0.9]
    popColors[ITsubset] = popColors['IT5A']
    popColors[SOMsubset] = popColors['SOM5A']
    popColors[PVsubset] = popColors['PV5A']
except:
    pass

def loadSimData(dataFolder, batchLabel, simLabel):
    ''' load sim file'''
    root = dataFolder+batchLabel+'/'
    sim,data,out = None, None, None
    if isinstance(simLabel, str): 
        filename = root+simLabel+'.json'
        print(filename)
        sim,data,out = utils.plotsFromFile(filename, raster=0, stats=0, rates=0, syncs=0, hist=0, psd=0, traces=0, grang=0, plotAll=0)
    
    return sim, data, out, root

def axisFontSize(ax, fontsize):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 


# ----------------------------------------------------------------
def fig_fIcurves():
    ''' f-I curves for exc and inh cells'''
    dataFolder = '../data/'
    batchLabel =  'v52_batch1' #'v54_batch1'
    loadAll = 1
    outFolder = '/u/salvadord/Work/m1grant/m1paper/supportFigs/'

    fontsiz = 16
    
    # exc pops
    epops=['IT2','IT4','IT5A','IT5B','PT5B','IT6','CT6']
    
    dfpop,dfpiv = ba.popRateAnalysis(dataFolder, batchLabel, loadAll, pars=['IClamp1_amp', 'IClamp1_pop'], vals = 'IClamp1_pop', plotLine=True, \
    query = "IClamp1_pop in ['IT2', 'IT4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6']")  #, plotLine=True)
    #utils.setPlotFormat(numColors = len(dfpiv.columns))
    colors = [popColors[pop] for pop in epops]
    dfpiv = dfpiv[epops]
    dfpiv.plot(marker='o', colors=colors)    
    ax=plt.gca()
    plt.xlabel('Somatic current injection (nA)', fontsize=fontsiz)
    plt.ylabel('Firing rate (Hz)', fontsize=fontsiz)
    plt.legend(epops, fontsize=fontsiz)
    axisFontSize(ax, fontsiz)
    #ax.get_legend().set_title('Cell type')
    #plt.setp(ax.get_legend().get_title(),fontweight='bold')

    plt.savefig(outFolder+'fIexc.png')
    plt.show()

    # inh pops
    ipops= ['PV2', 'SOM2']
    
    dfpop,dfpiv = ba.popRateAnalysis(dataFolder, batchLabel, loadAll, pars=['IClamp1_amp', 'IClamp1_pop'], vals = 'IClamp1_pop', plotLine=True, \
    query = "IClamp1_pop in ['PV2', 'SOM2']")  #, plotLine=True)
    #utils.setPlotFormat(numColors = len(dfpiv.columns))
    colors = [popColors[pop] for pop in ['PV6', 'SOM6']]
    dfpiv = dfpiv[ipops]
    dfpiv.plot(marker='o', colors=colors)  
    ax=plt.gca()
    plt.xlabel('Somatic current injection (nA)', fontsize=fontsiz)
    plt.ylabel('Firing rate (Hz)', fontsize=fontsiz)
    plt.legend(['PV','SOM'], fontsize=fontsiz)
    axisFontSize(ax, fontsiz)
    #ax.get_legend().set_title('Cell type', fontweight='bold')

    plt.savefig(outFolder+'fIinh.png')
    plt.show()


# ----------------------------------------------------------------
def fig_spont():
    ''' Figure spontaneous activity: 
        - raster plot of 3-4 sec 
        - traces 3-4 sec
        - histogram of log rates for IT,PT,CT,SOM,PV1; 50 sec 
        - boxplot of variability of log rates across 25 iseed x 5 sec
        - scatter of PT rate vs NCD, 25 iseed x 5 sec
        - PSTH of IT5B, PT5B upper, PT5B lower'''


    # ---------------------------------------------------------------------------------------------------------------
    # Config

    raster = 0
    traces = 0
    stats_boxplot = 1              # boxplot of rates
    stats_hist = 0        # histogram of rates (1 sim)
    stats_hist_multiple = 0   # histogram/stats of rates (N sims)
    stats_scatter = 0           # PT rate vs NCD (1 sim)
    stats_scatter_multiple = 0  # PT rate vs NCD (N sims)
    plotAll = 0


    dataFolder = '../data/'
    if  raster or traces:  # 2 sec N=1  
        batchLabel = 'v53_batch12' #'v53_batch12' 
        simLabel = 'v53_batch12_0_0_0' #'v53_batch12_0_0_0'
    elif stats_hist or stats_boxplot or stats_scatter: # 50 sec N=1
        batchLabel = 'v53_batch11'
        simLabel = 'v53_batch11_0_0'
    elif stats_hist_multiple or stats_scatter_multiple:  # 5 sec N=25
        loadAll = 1
        batchLabel = 'v53_batch13'
        simLabel =  ['v53_batch13_0_'+str(iseed)+'_'+str(iconn)+'_0' for iseed in range(5) for iconn in range(5)]

    sim, data, out, root = loadSimData(dataFolder, batchLabel, simLabel)


    plt.style.use('seaborn-ticks') 

    # ---------------------------------------------------------------------------------------------------------------
    # raster
    if raster or plotAll:
        timeRange = [2000, 4000] #[2000, 4000]
        include = allpops
        orderBy = ['pop', 'y']
        #filename = '%s%s_raster_%d_%d_%s.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
        fig1 = sim.analysis.plotRaster(include=include, timeRange=timeRange, labels='overlay', 
            popRates=0, orderInverse=True, lw=0, markerSize=3.5, marker='.', popColors=popColors, 
            showFig=0, saveFig=0, figSize=(8.5,10), orderBy=orderBy)# 
        ax = plt.gca()

        [i.set_linewidth(0.5) for i in ax.spines.values()] # make border thinner
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')  #remove ticks
        plt.ylabel(' ') #Neurons (ordered by NCD within each pop)')
        plt.xlabel(' ')
        
        plt.title('')
        filename='%s%s_raster_%d_%d_%s_test.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
        plt.savefig(filename, dpi=600)

        

    # ---------------------------------------------------------------------------------------------------------------
    # traces 
    if traces or plotAll:
        fontsiz = 20
        timeRange = [2000, 4000]
        # 6 sec cells: [50,19,393,799]; 2-sec cells: [393,447,579,19,104,214,1138,979,799]
        include = [3328, 5709] #[2900, 5234]
            # [(pop,50) for pop in ['IT2', 'IT4', 'IT5A', 'PT5B']] \
            #     + [('IT5A',x) for x in [393,447,579,19,104]] \
            #     + [('PT5B',x) for x in [393,447,579,19,104,214,1138,979,799]]
        #[('IT5A',50)] +[('PT5B',x) for x in [50,19, 393,447,579,19,104,214,1138,979,799]] #[19]] 
        colors = [popColors[p] for p in ['IT5A', 'PT5B']] #, 'CT6', 'S2', 'M2','PT5B', 'CT6', 'S2', 'M2' ]]

        fig4 = sim.analysis.plotTraces(include=include, timeRange=timeRange, colors=colors, 
            overlay=True, oneFigPer='trace', rerun=False, ylim=[-85, 30], axis='off', 
            figSize=(15,3.5), saveData=None, saveFig=0, showFig=0)


        # plt.ylabel('V (mV)',fontsize=fontsiz)
        # plt.xlabel('time (ms)', fontsize=fontsiz)
        #plt.figure(num=fig4.values()[0].number, figsize=(14,4))        
        #plt.subplots_adjust(right=0.9)
        #ax = plt.gca()
        #plt.setp(ax.get_xticklabels(),fontsize=fontsiz)
        
        plt.legend(['IT5A', 'PT5B'],fontsize=fontsiz, loc=2, bbox_to_anchor=(1.05, 1))
        plt.title('')
        plt.tight_layout()
        plt.savefig('%s%s_traces_%d_%d.png'%(root, simLabel, timeRange[0], timeRange[1]), dpi=200)

        

    # ---------------------------------------------------------------------------------------------------------------
    # stats exc
    if stats_boxplot or plotAll:        
        fontsiz = 20
        timeRange = [1000, 51000]
        include = excpops+[SOMsubset,PVsubset]
        #include = [ITsubset, 'PT5B', 'CT6', SOMsubset, PVsubset] #reverse
        labels = excpops+ ['SOM L2-6','PV L2-6'] #reverse
        filename = root+simLabel+'_%d_%d_exc'%(timeRange[0], timeRange[1])
        xlim = [0,145] # ok to cut off max and flyiers (Bill19)
        fig1,data1 = sim.analysis.plotSpikeStats(include=include, figSize=(8,4), timeRange=timeRange, xlim=xlim,
            stats = ['rate'], legendLabels=labels, fontSize = fontsiz, popColors=popColors, showFig=0, dpi=300, saveFig=filename)
        xlim = [0,12]  

        import IPython; IPython.embed()

        # replot with exp data 
        expData = {}
        
        ## PV rates from Este18 
        from scipy.io import loadmat
        matData = loadmat('../data/Este18/data.mat')
        expData['PV'] = matData['baseline'][0][0][2][0]

        statDataExp = [[]]*len(excpops)+[[], expData['PV']]

        statData = data1['statData']
        colors = [popColors[p] for p in include]

        plt.figure(figsize=(10,12))
        meanpointprops = dict(marker = (5, 1, 0), markeredgecolor = 'black', markerfacecolor = 'white')
        
        bp=plt.boxplot(statData[::-1], positions=np.array(range(len(statData)))*2.0+0.4,labels=labels[::-1], notch=False, sym='k+', meanprops=meanpointprops,  
                    whis=1.5, widths=0.6, vert=False, showmeans=True, patch_artist=True)  #labels[::-1]

        bpexp=plt.boxplot(statDataExp[::-1], positions=np.array(range(len(statDataExp)))*2.0-0.4, labels=labels[::-1], notch=False, sym='k+', meanprops=meanpointprops,  whis=1.5, widths=0.6, vert=False, showmeans=True, patch_artist=True)  #labels[::-1]


        plt.xlabel('Rate (Hz)', fontsize=fontsiz)
        plt.ylabel('Population', fontsize = fontsiz)
        plt.subplots_adjust(left=0.15,right=0.95, top=0.9, bottom=0.1)

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c='#999999', linewidth=10, label='Experiment')
        plt.legend()        

        icolor=0
        borderColor = 'k'
        for i in range(0, len(bp['boxes'])):
            icolor = i
            bp['boxes'][i].set_facecolor(colors[::-1][icolor])
            bp['boxes'][i].set_linewidth(2)
            # we have two whiskers!
            bp['whiskers'][i*2].set_color(borderColor)
            bp['whiskers'][i*2 + 1].set_color(borderColor)
            bp['whiskers'][i*2].set_linewidth(2)
            bp['whiskers'][i*2 + 1].set_linewidth(2)
            bp['medians'][i].set_color(borderColor)
            bp['medians'][i].set_linewidth(3)
            for c in bp['caps']:
                c.set_color(borderColor)
                c.set_linewidth(2)

        borderColor = 'k'
        expColor = [0.7,0.7,0.7]
        for i in range(0, len(bp['boxes'])):
            icolor = i
            bpexp['boxes'][i].set_facecolor(expColor)
            bpexp['boxes'][i].set_linewidth(2)
            # we have two whiskers!
            bpexp['whiskers'][i*2].set_color(borderColor)
            bpexp['whiskers'][i*2 + 1].set_color(borderColor)
            bpexp['whiskers'][i*2].set_linewidth(2)
            bpexp['whiskers'][i*2 + 1].set_linewidth(2)
            bpexp['medians'][i].set_color(borderColor)
            bpexp['medians'][i].set_linewidth(3)
            for c in bpexp['caps']:
                c.set_color(borderColor)
                c.set_linewidth(2)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', direction='out')
        ax.grid(axis='x', color="0.9", linestyle='-', linewidth=1)
        ax.set_axisbelow(True)
        if xlim: ax.set_xlim(xlim)

        ticks = labels
        plt.yticks(range(0, len(ticks) * 2, 2), ticks[::-1])
        plt.ylim(-2, len(ticks)*2)
        


        # fig2,data2 = sim.analysis.plotSpikeStats(include=include, figSize=(8,4), timeRange=timeRange, xlim=xlim,
        #     stats = ['isicv'], legendLabels=labels, fontSize = fontsiz, popColors=popColors, showFig=0, dpi=300, saveFig=filename)

        #plt.savefig(filename, dpi=300)


    # ---------------------------------------------------------------------------------------------------------------
    # stats inh
    if 0: #stats_boxplot or plotAll:
        timeRange = [1000, 5000]
        include = inhpops
        filename = root+simLabel+'_%d_%d_inh'%(timeRange[0], timeRange[1])
        #fig1 = sim.analysis.plotSpikeStats(include=include, figSize=(4,8), timeRange=timeRange, 
        #    stats = ['rate', 'isicv'], fontsize = fontsiz, popColors=popColors, showFig=0, 
        #    saveFig=filename)
        #plt.savefig(filename, dpi=300)


    # ---------------------------------------------------------------------------------------------------------------
    # stats_hist exc - Lin or Log
    if stats_hist or plotAll:
        fontsiz = 18
        timeRange = [1000, 51000]
        include = ['PT5B', ITsubset] #['CT6', 'PT5B', ITsubset]
        histmin = 0.1  
        bins = 20
        lognorm = 1
        if lognorm:
            histlogx = 1
            histlogy = 1
            density = 1
        else:
            histlogx = 0
            histlogy = 0
            density = 0

        filename = root+simLabel+'_%d_%d_exc_log_%d_%d_dens_%d_bins_%d'%(timeRange[0], timeRange[1], histlogx, histmin, density, bins)
        fig, data = sim.analysis.plotSpikeStats(include=include, graphType='histogram', figSize=(9,5), 
            density=density, histlogx=histlogx, histlogy=histlogy, histmin=histmin, bins=bins, timeRange=timeRange, stats = ['rate'], normfit=0,
            fontsize = fontsiz+8, popColors=popColors, legendLabels=['PT', 'IT'], histShading=1, dpi=200, showFig=0, saveFig=filename)

        statData, gidsData, ynormsData = data['statData'], data['gidsData'], data['ynormsData']


        # calculate log mean and std of each rate pops for each sim
        def logmean(data):
            #return np.mean(np.log(data))
            return np.log10(np.mean(data))

        def logstd(data, logmean):
            #return np.sqrt(np.mean(np.power(np.log(data) - logmean, 2)))
            return np.log10(np.std(data))

        stat_tests = True
        if stat_tests:
            import scipy
            for ipop in range(len(include)):
                nonzeroData = statData[ipop]
                nonzeroData = [x for x in nonzeroData if x > histmin]
                logData = np.log10(nonzeroData)
                logMeanValue = logmean(nonzeroData)
                logStdValue = logstd(nonzeroData, logMeanValue) # https://en.wikipedia.org/wiki/Log-normal_distribution#Maximum_likelihood_estimation_of_parameters
                logVarValue = logStdValue ** 2
                logSkew = scipy.stats.skew(logData)
                logKurtosis = scipy.stats.kurtosis(logData)
                N = len(nonzeroData)
                SESkew = np.sqrt((6*N*(N-1))/((N-2)*(N+1)*(N+3)))
                ratioSkew = logSkew / SESkew
                SEKurtosis = 2*SESkew*np.sqrt((N*N-1)/((N-3)*(N+5)))
                ratioKurtosis = logSkew / SEKurtosis
                print('\n-----\n%s: lognormal mean=%f, std=%f, var=%f, skewness=%f, kurtosis=%f, ratioSkew=%f, ratioKurtosis=%f\n'
                    % (include[-(ipop + 1)], logMeanValue, logStdValue, logVarValue, logSkew, logKurtosis, ratioSkew,  ratioKurtosis))

                # interpret
                alpha = 2.58
                if abs(ratioSkew) < alpha:
                    print('Skew ratio: Sample looks Gaussian (fail to reject H0)')
                else:
                    print('Skew ratio: Sample does not look Gaussian (reject H0)')

                # interpret
                alpha = 2.58
                if abs(ratioKurtosis) < alpha:
                    print('Kurtosis ratio: Sample looks Gaussian (fail to reject H0)')
                else:
                    print('Kurtosis ratio: Sample does not look Gaussian (reject H0)')

                stat, p = scipy.stats.shapiro(logData)
                print('Shapiro-Wilk Test: Statistics=%.3f, p=%g' % (stat, p))
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    print('Shapiro-Wilk test: Sample looks Gaussian (fail to reject H0)')
                else:
                    print('Shapiro-Wilk test: Sample does not look Gaussian (reject H0)')

                # boxplot
                plt.boxplot(logData)
                median = np.median(logData)
                q25 = np.percentile(logData, 25)
                q75 = np.percentile(logData, 75)
                q0 = np.percentile(logData, 0)
                q100 = np.percentile(logData, 100)
                minV = np.min(logData)
                maxV = np.max(logData)
                print('%s: min, Q25, median, Q75, max, IQR = %f, %f, %f, %f, %f, %f\n' % (include[-(ipop + 1)], minV, q25, median, q75, maxV, q75-q25))
                print('\nIQ symmetry: %f, %f ; Tails larger: %f, %f'%(median-q25, q75-median, q25-q0, q100-q75))
                (mu, sigma) = scipy.stats.norm.fit(stat)
                print('%s: mu, sigma = %f, %f \n' % (include[-(ipop + 1)], mu, sigma))

        plt.ylim(0,0.1)
        plt.savefig(filename, dpi=300)

        

    # ---------------------------------------------------------------------------------------------------------------
    # stats_hist inh - Lin or Log
    if 0: #stats_hist or plotAll:
        timeRange = [1000, 51000]
        include = [PVsubset,SOMsubset]
        histmin = 0.05
        bins = 20
        lognorm = 0
        if lognorm:
            histlogx = 1
            histlogy = 1
            density = 1
        else:
            histlogx = 0
            histlogy = 0
            density = 0


        filename = root+simLabel+'_%d_%d_inh_log_%d_%d_dens_%d_bins_%d'%(timeRange[0], timeRange[1], histlogx, histmin, density, bins)
        fig1, statData, gidsData, ynormsData = sim.analysis.plotSpikeStats(include=include, graphType='histogram', figSize=(7,6), 
            density=density, histlogx=histlogx,  histlogy=histlogy, histmin=histmin, bins=bins, timeRange=timeRange, stats = ['rate'], normfit=1,
            fontsize = fontsiz, popColors=popColors, legendLabels=['PV', 'SOM'], histShading=1, dpi=200, showFig=0, saveFig=filename)


    # ---------------------------------------------------------------------------------------------------------------
    # stats_hist_multiple exc
    if stats_hist_multiple or plotAll:
        from operator import add
        import scipy.stats
        from matplotlib import mlab

        root = dataFolder+batchLabel+'/'
        timeRange = [500, 2000]
        include = ['CT6', 'PT5B', ITsubset]

        if not loadAll:
            # load data from all sim files
            statDataAll = [[]]*len(simLabel)
            for isim,simLab in enumerate(simLabel):  # get data from each sim 
                filename = root+simLab+'.json'
                print(filename)
                sim,data,out = utils.plotsFromFile(filename, raster=0, stats=0, rates=0, syncs=0, hist=0, psd=0, traces=0, grang=0, plotAll=0)
                _, data = sim.analysis.plotSpikeStats(include=include, graphType='none', includeRate0=1, timeRange=timeRange, stats=['rate'], showFig=0, saveFig=0)
                statDataAll[isim], gidsData, ynormsData = data['statData'], data['gidsData'], data['ynormsData']

            # save
            statData = {'statDataAll': statDataAll,  'gidsData': gidsData, 'ynormsData': ynormsData}
            with open(root+'%s_statDataAll_hist.json'%(simLabel[0][:-2]), 'w') as fileObj:
                json.dump(statData, fileObj)

            # load All        
        else:
            with open(root+'%s_statDataAll_hist.json'%(simLabel[0][:-2]), 'r') as fileObj:
                statData = json.load(fileObj)
            filename = root+simLabel[0]+'.json'
            sim,data,out = utils.plotsFromFile(filename, raster=0, stats=0, rates=0, syncs=0, hist=0, psd=0, traces=0, grang=0, plotAll=0)
          

        # combine 
        statDataAll = statData['statDataAll']
        statDataAvg = statData['statDataAll'][0]
        gidsData = statData['gidsData']
        ynormsData = statData['ynormsData']
        
        # calculate log mean and std of each rate pops for each sim
        means = [[]]*len(include)
        stds = [[]]*len(include)
        filename = root+batchLabel+'N_%d_%d_exc_log.stats'%(timeRange[0], timeRange[1])
        f = open(filename, 'w')

        def logmean(data):
            return np.mean(np.log(data))
            #return np.mean(data)

        def logstd(data, logmean):
            return np.sqrt(np.mean(np.power(np.log(data) - logmean, 2)))
            #return np.std(data)

        for ipop in range(len(include)):
            for isim in range(len(simLabel)):
                nonzeroData = statDataAll[isim][ipop]
                nonzeroData = [x for x in nonzeroData if x>0.0]
                logMeanValue = logmean(nonzeroData)
                means[ipop].append(logMeanValue) # calculate mean of logs
                stds[ipop].append(logstd(nonzeroData, logMeanValue)) # https://en.wikipedia.org/wiki/Log-normal_distribution#Maximum_likelihood_estimation_of_parameters
            s = '\n\n%s:' % (str(include[ipop]))
            print(s)
            f.write(s)
            for statLabel, stat in zip(['Means', 'Stds'], [means, stds]):
                median = np.median(stat)
                q25 = np.percentile(stat, 25)
                q75 = np.percentile(stat, 75)
                minV = np.min(stat)
                maxV = np.max(stat)
                s = '%s: min, Q25, median, Q75, max, IQR = %f, %f, %f, %f, %f, %f\n' % (statLabel, minV, q25, median, q75, maxV, q75-q25)
                print(s)
                f.write(s)
                W, p = scipy.stats.shapiro(stat)
                (mu, sigma) = scipy.stats.norm.fit(stat)
                s = '%s: mu, sigma, W, p = %f, %f, %f, %f\n' % (statLabel, mu, sigma, W, p)
                print(s)
                f.write(s)
        f.close()


                    # # plot
                    # n, bins,_ = plt.hist(stat)
                    # y = mlab.normpdf(bins, mu, sigma)
                    # l = plt.plot(bins, y, 'r--', linewidth=2)

        # boxplot of mean + std for log distribution of rates for each pop
        #plt.boxplot([means,stds])

        # plotting
        plot = 1
        avg = 0
        if plot:
            if avg:
                for ipop in range(len(include)):
                    for isim in range(len(simLabel)-1):
                        statDataAvg[ipop] = map(add, statDataAvg[ipop], statDataAll[isim+1][ipop])                
                    statDataAvg[ipop] = [x / len(simLabel) for x in statDataAvg[ipop]]
            else:
                for ipop in range(len(include)):
                    for isim in range(len(simLabel)-1):
                        statDataAvg[ipop].extend(statDataAll[isim+1][ipop]) 

            fontsiz = 18
            timeRange = [500, 2000]
            include = ['CT6', 'PT5B', ITsubset]
            filename = root+batchLabel+'N_%d_%d_exc_log'%(timeRange[0], timeRange[1])
            statDataIn = {'rate': {'statData': statDataAvg, 'gidsData': gidsData, 'ynormsData': ynormsData}}
            fig1 = sim.analysis.plotSpikeStats(include=include, statDataIn=statDataIn, graphType='histogram', figSize=(7,6), density=0, histmin=0.1, histlogx=1, bins=50,
                timeRange=timeRange, stats = ['rate'], fontsize = fontsiz, popColors=popColors, legendLabels=['CT', 'PT', 'IT'],
                dpi=200, showFig=0, saveFig=filename)


    # ---------------------------------------------------------------------------------------------------------------    
    # stats_scatter PT rate vs NCD
    if stats_scatter or plotAll:
        fontsiz = 18
        timeRange = [1000, 6000]
        include = ['PT5B']
        filename = root+simLabel+'_%d_%d_exc'%(timeRange[0], timeRange[1])
        fig1 = sim.analysis.plotSpikeStats(include=include, graphType='scatter', figSize=(7,6),  
            timeRange=timeRange, stats=['rate'], bins=10,fontsize=fontsiz, popColors=popColors, 
            dpi=200, showFig=0, saveFig=0)
        ax = plt.gca()
        #plt.title('PT5B firing rates as a function of NCD', fontsize=fontsiz)
        plt.xlabel('Normalized cortical depth (NCD)', fontsize=fontsiz)
        plt.ylabel('Avg firing rate (Hz)')
        ax.legend_.remove()

        #plt.title('')
        filename='%s%s_exc_spikeStat_scatter_rate_%d_%d.png'%(root, simLabel[0], timeRange[0], timeRange[1])
        plt.savefig(filename, dpi=200)


    # ---------------------------------------------------------------------------------------------------------------
    # stats_scatter_multiple PT rate vs NCD
    if stats_scatter_multiple or plotAll:
        from operator import add
        root = dataFolder+batchLabel+'/'
        timeRange = [1000, 6000]
        include = ['PT5B']

        if not loadAll:
            print('Loading individual sim file ...')
            # load data from all sim files
            statDataAll = [[]]*len(simLabel)
            for isim,simLab in enumerate(simLabel):  # get data from each sim 
                filename = root+simLab+'.json'
                print('\n')
                print(filename)
                sim,data,out = utils.plotsFromFile(filename, raster=0, stats=0, rates=0, syncs=0, hist=0, psd=0, traces=0, grang=0, plotAll=0)
                _, statDataAll[isim], gidsData, ynormsData  = sim.analysis.plotSpikeStats(include=include, graphType='scatter', includeRate0=1, timeRange=timeRange, stats=['rate'], showFig=0, saveFig=0)

            # save
            statData = {'statDataAll': statDataAll,  'gidsData': gidsData, 'ynormsData': ynormsData}
            with open(root+'%s_statDataAll_scatter.json'%(simLabel[0][:-2]), 'w') as fileObj:
                json.dump(statData, fileObj)

            # load All        
        else:
            with open(root+'%s_statDataAll_scatter.json'%(simLabel[0][:-2]), 'r') as fileObj:
                statData = json.load(fileObj)
            filename = root+simLabel[0]+'.json'
            sim,data,out = utils.plotsFromFile(filename, raster=0, stats=0, rates=0, syncs=0, hist=0, psd=0, traces=0, grang=0, plotAll=0)
          
        # combine 
        print('\nMerging data from different sims...')

        statDataAll = statData['statDataAll']
        statDataAvg = statData['statDataAll'][0]
        gidsData = statData['gidsData']
        ynormsData = statData['ynormsData']
        
        avg = 0
        if avg:
            for ipop in range(len(include)):
                for isim in range(len(simLabel)-1):
                    statDataAvg[ipop] = map(add, statDataAvg[ipop], statDataAll[isim+1][ipop])                
                statDataAvg[ipop] = [x / len(simLabel) for x in statDataAvg[ipop]]
        else:
            for ipop in range(len(include)):
                ynormsData[ipop] = ynormsData[ipop]*len(statDataAll)
                for isim in range(len(simLabel)-1):
                    statDataAvg[ipop].extend(statDataAll[isim+1][ipop]) 

        print('Plotting rate vs NCD scatter ...')
        fontsiz = 20
        include = ['PT5B']
        filename = root+batchLabel+'_%d_%d_exc'%(timeRange[0], timeRange[1])
        statDataIn = {'rate': {'statData': statDataAvg, 'gidsData': gidsData, 'ynormsData': ynormsData}}
        fig1 = sim.analysis.plotSpikeStats(include=include,  statDataIn=statDataIn, graphType='scatter', figSize=(9,5),  
            timeRange=timeRange, stats=['rate'], bins=20, fontSize=fontsiz, popColors=popColors, differentColor=[1435, [0/255.0,215/255.0,255/255.0]],
            dpi=200, showFig=0, saveFig=0)
        ax = plt.gca()
        #plt.title('PT5B firing rates as a function of NCD', fontsize=fontsiz)
        plt.xlabel('Normalized cortical depth (NCD)', fontsize=fontsiz)
        plt.ylabel('Avg firing rate (Hz)', fontsize=fontsiz)
        ax.legend_.remove()

        #plt.title('')
        filename='%s%s_exc_spikeStat_scatter_N_rate_%d_%d.png'%(root, batchLabel, timeRange[0], timeRange[1])
        plt.savefig(filename, dpi=200)


    return sim


# ----------------------------------------------------------------
def fig_osc_lfp():
    ''' Figure LFP filt and spectrogrma: 
        - LFP signal + filtered slow+gamma
        - LFP Spectrogram
        '''

    # ---------------------------------------------------------------------------------------------------------------
    # options

    lfp = 0
    lfp_filtered = 0
    lfp_spectrogram = 0
    lfp_cfc = 0
    lfp_cfc_stats = 1
    plotAll = 0
   
    dataFolder = '../data/'
    
    # single sim
    # batchLabel = 'v53_batch12'
    # simLabels = ['v53_batch12_0_0_0']
    
    fontsiz = 16
    # N=25
    batchLabel = 'v53_batch13'
    simLabels =  ['v53_batch13_0_'+str(iseed)+'_'+str(iconn)+'_0' for iseed in range(1) for iconn in range(1)]
    
    saveToFile = 0
    loadFromFile = 1
    dataSave = []

    if lfp or lfp_filtered or lfp_spectrogram or lfp_cfc or plotAll:

        root = dataFolder + batchLabel + '/'

        for simLabel in simLabels:

            if not loadFromFile:
                sim, data, out, root = loadSimData(dataFolder, batchLabel, simLabel)

            plt.style.use('seaborn-ticks') 

            # ---------------------------------------------------------------------------------------------------------------
            # LFP signal
            if lfp or plotAll:
                
                electrodes = [6] #range(11) #+avg [1,4,6,8,10]
                ''' electrode locations = range(200,1300,100)
                0=200 (L23), 1=300 (L23), 2=400 (L23/4 border), 3=500 (L4/5A border), 4=600 (L5A), 5=700 (L5Bu), 
                6=800 (L5Bu), 7=900 (L5Bm), 8=1000 (L5Bl), 9=1100 (L6), 10=1200 (L6), 11=1300 (L6)
                '''
                # time series (inset - 200ms)
                timeRange = [3500, 4000] #3700]
                filtFreq = 200 #[100, 250]
                plots = ['timeSeries']
                #filename= '%s%s_lfp_timeSeries_%d_%d_filt_%d_%d.png' % (root, simLabel, timeRange[0], timeRange[1], filtFreq[0], filtFreq[1])
                filename= '%s%s_lfp_timeSeries_%d_%d_filt_%d.png' % (root, simLabel, timeRange[0], timeRange[1], filtFreq)
                fig4 = sim.analysis.plotLFP(plots=plots, electrodes=electrodes, timeRange=timeRange, figSize=(8,4), overlay=True, 
                    separation=1.2, colors=[[0,0,0]]*11, filtFreq=filtFreq, dpi=300, saveFig=False, showFig=False) 
                plt.suptitle('')
                plt.savefig(filename,dpi=300)

                # time series full - 4 sec
                timeRange = [1000, 5000]
                plots = ['timeSeries']
                #filename= '%s%s_lfp_timeSeries_%d_%d_filt_%d_%d.png' % (root, simLabel, timeRange[0], timeRange[1], filtFreq[0], filtFreq[1])
                filename= '%s%s_lfp_timeSeries_%d_%d_filt_%d.png' % (root, simLabel, timeRange[0], timeRange[1], filtFreq)
                fig4, figdata = sim.analysis.plotLFP(plots=plots, electrodes=electrodes, timeRange=timeRange, figSize=(8,4), overlay=True, 
                    separation=1.5, dpi=300, colors=[[0,0,0]]*11, filtFreq=filtFreq, saveFig=0, showFig=False) 
                lfpPlot = figdata['lfpPlot']
                ydisp = figdata['ydisp']
                t = figdata['t']
                color = 'black'
                lw = 0.5

                plt.figure(figsize=(8,4))
                fontsiz = 16
                plt.plot(t, -lfpPlot, color=color, linewidth=lw)
                # plt.plot([3500, 3500], [0,1], 'r')
                # plt.plot([3700, 3700], [0,1], 'r') 
                ax = plt.gca()        
                ax.invert_yaxis()
                plt.axis('off')
                plt.xlabel('time (ms)', fontsize=fontsiz)
                #plt.subplots_adjust(bottom=0.1, top=1.0, right=1.0)

                # calculate scalebar size and add scalebar
                scaley = 1000.0  # values in mV but want to convert to uV
                sizey = 100/scaley
                labely = '%.3g $\mu$V'%(sizey*scaley)#)[1:]
                add_scalebar(ax, hidey=True, matchy=False, hidex=True, matchx=True, sizex=None, sizey=-sizey, labely=labely, unitsy='$\mu$V', scaley=scaley, 
                        unitsx='ms', loc=9, pad=0.5, borderpad=0.5, sep=3, prop=None, barcolor="black", barwidth=2)
                
                plt.title('LFP 0-200 Hz', fontsize=fontsiz, fontweight='bold')
                plt.savefig(filename,dpi=300)


            # ---------------------------------------------------------------------------------------------------------------
            # LFP filtered
            if lfp_filtered or plotAll:
                
                electrodes = [6]

                # get lfp data - 0-4Hz
                timeRange = [1000, 5000]
                plots = ['timeSeries']
                filtFreq = 4
                filename= '%s%s_lfp_timeSeries_filtered_%d_%d_0_%dHz.png' % (root, simLabel, timeRange[0], timeRange[1], filtFreq)
                fig4,figdata = sim.analysis.plotLFP(plots=plots, electrodes=electrodes, timeRange=timeRange, figSize=(8,4), overlay=True, 
                    separation=1.5, dpi=400, filtFreq=filtFreq, detrend=0, norm=0, colors=[[0,0,0]]*11,  saveFig=False, showFig=False) 
                lfpPlot = figdata['lfpPlot']
                ydisp = figdata['ydisp']
                t = figdata['t']
                color = 'black'
                lw = 0.5
                plt.figure(figsize=(8,2.2))
                plt.plot(t, -lfpPlot, color=color, linewidth=lw)
                ax = plt.gca()        
                ax.invert_yaxis()
                plt.axis('off')
                plt.xlabel('time (ms)', fontsize=fontsiz)
                scaley = 1000.0  # values in mV but want to convert to uV
                sizey = 100/scaley
                labely = '%.3g $\mu$V'%(sizey*scaley)#)[1:]
                add_scalebar(ax, hidey=True, matchy=False, hidex=True, matchx=True, sizex=None, sizey=-sizey, labely=labely, unitsy='$\mu$V', scaley=scaley, 
                        unitsx='ms', loc=1, pad=0.5, borderpad=0.5, sep=3, prop=None, barcolor="black", barwidth=2)
                
                plt.title('LFP 0-4 Hz', fontsize=fontsiz, fontweight='bold')
                plt.savefig(filename,dpi=300)


                # get lfp data - 30-40Hz
                timeRange = [1000, 5000]
                plots = ['timeSeries']
                filename= '%s%s_lfp_timeSeries_Filtered_%d_%d_30_40hz.png' % (root, simLabel, timeRange[0], timeRange[1])
                fig4,figdata = sim.analysis.plotLFP(plots=plots, electrodes=electrodes, timeRange=timeRange, figSize=(8,4), overlay=True, 
                    separation=1.5, dpi=400, filtFreq=[30,40], detrend=0, normSignal=1, colors=[[0,0,0]]*11,  saveFig=filename, showFig=False) 
                lfpPlot = figdata['lfpPlot']
                ydisp = figdata['ydisp']
                t = figdata['t']
                color = 'black'
                lw = 0.5
                plt.figure(figsize=(8,2.2))
                fontsiz = 16
                plt.plot(t, -lfpPlot, color=color, linewidth=lw)
                ax = plt.gca()        
                ax.invert_yaxis()
                plt.axis('off')
                plt.xlabel('time (ms)', fontsize=fontsiz)
                scaley = 1000.0  # values in mV but want to convert to uV
                sizey = 100/scaley
                labely = '%.3g $\mu$V'%(sizey*scaley)#)[1:]
                add_scalebar(ax, hidey=True, matchy=False, hidex=True, matchx=True, sizex=None, sizey=-sizey, labely=labely, unitsy='$\mu$V', scaley=scaley, 
                        unitsx='ms', loc=1, pad=-1.0, borderpad=0.5, sep=3, prop=None, barcolor="black", barwidth=2)
                
                plt.title('LFP 30-40 Hz', fontsize=fontsiz, fontweight='bold')
                plt.savefig(filename,dpi=300)


            # ---------------------------------------------------------------------------------------------------------------
            # LFP Spectrogram
            if lfp_spectrogram or plotAll:
                # spectrogram
                timeRange = [1000, 5000]
                electrodes = [6]
                plots = ['spectrogram'] #['PSD','timeSeries', 'spectrogram']
                filtFreq = 200

                fig4 = sim.analysis.plotLFP(plots=plots, electrodes=electrodes, timeRange=timeRange, figSize=(8,4), overlay=True, maxFreq=80,
                    normSpec=True,  dpi=300, saveFig=0, showFig=False) 

                ax = plt.gca()
                plt.title('')
                plt.suptitle('')
                #plt.suptitle('LFP spectrogram', fontsize=fontsiz,  fontweight='bold')
                ax.set_xticklabels([1,2,3,4,5])
                ax.set_xticks([1000, 2000, 3000, 4000, 5000])
                plt.setp(ax.get_xticklabels(),fontsize=fontsiz-2)
                plt.setp(ax.get_yticklabels(),fontsize=fontsiz-2)
                plt.ylabel('Frequency (Hz)',fontsize=fontsiz)
                plt.xlabel('Time (s)',fontsize=fontsiz)
                plt.subplots_adjust(bottom=0.25, top=0.9, right=1.0, left=0.1)
                plt.savefig('%s%s_lfp_spec_%d_%d_filt_%d.png' % (root, simLabel, timeRange[0], timeRange[1], filtFreq), dpi=300)

                # # for nfft in [4, 8, 12]:
                # #     for overlap in [0.25, 0.5, 0.75]:
                # #         for nperseg in [1.03125, 1.1, 1.2, 1.3]:
                # #             filename='%s%s_spect_%d_%d_%d_%f_%f_.png' % (root, simLabel, timeRange[0], timeRange[1], nfft, overlap, nperseg)
                # #             print(filename)
                # #             try:
                # #                 fig4 = sim.analysis.plotLFP(plots=plots, electrodes=electrodes, timeRange=timeRange, figSize=(8,8), overlay=True, 
                # #                     NFFT=int(32*20*nfft), noverlap=int(32*20*nfft*overlap), nperseg=int((32*20*nfft*overlap)*nperseg), separation=1.5, logy=[10, 20, 40, 80],  dpi=200, saveFig=filename, showFig=False) 
                # #             except:
                # #                 pass
    
            # ---------------------------------------------------------------------------------------------------------------
            # LFP cross freq coupling (CFC)
            if lfp_cfc or plotAll:
                
                from neuron import h

                # params
                filtFreq = 200
                timeRange = [1000, 5000]
                electrodes = [6] #range(11)
                from_t = 1
                to_t = 5  # in secs,1-5 sec
                f_phaseMin = 1
                f_phaseMax = 4
                f_ampMin = 20
                f_ampMax = 80  
                phaseStep = 1
                phase_bandWidth = 1
                ampStep = 2
                amp_bandWidth = ampStep*2
                nBins = 18
                NShuffle = 50
                Z_score = False


                plotFig = 1

                if loadFromFile:
                    with open('%s%s_lfp_phase-amp_cfc_stats_data_%d_%d_ampStep_%d.pkl' % (root, batchLabel, timeRange[0], timeRange[1], ampStep), 'rb') as f:
                        elecData = pickle.load(f)[0]
                    phaseFreq = range(f_phaseMin, f_phaseMax+1, phaseStep)
                    ampFreq = range(f_ampMin, f_ampMax+1, ampStep)
                else:
                    # preprocess signal
                    import modindex
                    fs = 1000.0/sim.cfg.recordStep 
                    lfpData = np.array(sim.allSimData['LFP']) #[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]

                    if filtFreq:
                        from scipy import signal
                        
                        nyquist = fs / 2.0
                        filtOrder = 3
                        Wn = filtFreq/nyquist
                        b, a = signal.butter(filtOrder, Wn)
                        for i in range(lfpData.shape[1]):
                            lfpData[:,i] = signal.filtfilt(b, a, lfpData[:,i])

                    elecData = {}
                
                if plotFig:
                    figsize = (8, 8)
                    fontsiz=20
                    fig = plt.figure(figsize=figsize)

                for ielec, elec in enumerate(electrodes):
                    if loadFromFile:
                        modArr, meanArr, stdArr, zscoreArr = [elecData['elec %d'%(elec)][s] for s in ['modArr', 'meanArr', 'stdArr', 'zscoreArr']]
                    else:  
                        # prepare signal
                        raw_signal = h.Vector(lfpData[:, elec])
                        
                        # calculate modulation index
                        phaseFreq, ampFreq, modArr, meanArr, stdArr, zscoreArr = modindex.modIndArr (raw_signal, fs, from_t, to_t, f_phaseMin, f_phaseMax, f_ampMin, f_ampMax,phaseStep, phase_bandWidth, ampStep, amp_bandWidth, nBins, NShuffle, Z_score )

                        # store data for each electrode
                        elecData['elec %d'%(elec)] = {'modArr': modArr, 'zscoreArr': zscoreArr, 'meanArr': meanArr, 'stdArr': stdArr}

                    # plot figure                
                    if plotFig:
                        #plt.subplot(2, np.ceil(len(electrodes) / 2.), ielec + 1)
                        #plt.title('Electrode %d' % (elec), fontsize=fontsiz)
                        if Z_score:
                            plt.contourf(phaseFreq, ampFreq, zscoreArr, cmap = plt.get_cmap('viridis'))
                            cbar = plt.colorbar()
                            cbar.set_label(label='Shuffle test z-score', fontsize=fontsiz)
                            cb.ax.tick_params(labelsize=fontsiz)
                        else:
                            plt.contourf(phaseFreq, ampFreq, modArr, cmap = plt.get_cmap('viridis'))
                            cbar = plt.colorbar()
                            cbar.set_label(label='Modulation index', fontsize=fontsiz)
                            cbar.ax.tick_params(labelsize=fontsiz)
                        plt.axis([np.array(phaseFreq).min(), np.array(phaseFreq).max(), np.array(ampFreq).min(), np.array(ampFreq).max()])
                        
                        ax = plt.gca()
                        ax.set_xticklabels([1,2,3,4])
                        ax.set_xticks([1,2,3,4])
                        for tick in ax.xaxis.get_major_ticks():
                            tick.label.set_fontsize(fontsiz) 
                        for tick in ax.yaxis.get_major_ticks():
                            tick.label.set_fontsize(fontsiz)
                        plt.xlabel('Frequency for phase', fontsize=fontsiz)
                        plt.ylabel('Frequency for amplitude', fontsize=fontsiz)
                
                # save multiple plot fig
                if plotFig:
                    plt.tight_layout()
                    plt.savefig('%s%s_lfp_phase-amp_cfc_%d_%d_ampStep_%d_modArr.png' % (root, simLabel, timeRange[0], timeRange[1], ampStep), dpi=300)
                        
                # save data
                if saveToFile:
                    dataSave.append(elecData)
                    with open('%s%s_lfp_phase-amp_cfc_stats_data_%d_%d_ampStep_%d.pkl' % (root, batchLabel, timeRange[0], timeRange[1], ampStep), 'wb') as f:
                        pickle.dump(dataSave, f)


    # ---------------------------------------------------------------------------------------------------------------
    # LFP CFC stats
    elif lfp_cfc_stats:

        root = dataFolder + batchLabel + '/'
        fontsiz = 14
        figsiz = (8,4)
        timeRange = [1000, 5000]
        nseeds = 25
        includeLabels = ['elec 6']

        f_phaseMin = 1
        f_phaseMax = 4
        f_ampMin = 20
        f_ampMax = 80  
        phaseStep = 1
        ampStep = 2

        plotBoxplot = True

        # load and preprocess data
        with open('%s%s_lfp_phase-amp_cfc_stats_data_%d_%d_ampStep_%d.pkl' % (root, batchLabel, timeRange[0], timeRange[1], ampStep), 'rb') as f:
            data = pickle.load(f)
        
        allData = {}
        maxMods = {}
        maxZscores = {}
        maxPhaseFreqs = {}
        maxAmpFreqs = {}

        #import IPython; IPython.embed()

        electrodes = [k for k in data[0].keys() if k in includeLabels] 

        phaseFreqs = range(f_phaseMin, f_phaseMax+1, phaseStep)
        ampFreqs = range(f_ampMin, f_ampMax+1, ampStep)

        for k in electrodes:
            allData[k] = []
            maxMods[k] = []
            maxZscores[k] = []
            maxPhaseFreqs[k] = []
            maxAmpFreqs[k] = []

            print('Analyzing %s' % (k))
            for iseed in range(nseeds):
                # group data by projection
                cfcMatrix = data[iseed][k]['modArr']
                zscoreMatrix = data[iseed][k]['zscoreArr']
                allData[k].append(cfcMatrix)
                maxMods[k].append(np.max(cfcMatrix))
                (i, j) = np.unravel_index(cfcMatrix.argmax(), cfcMatrix.shape)
                maxZscores[k].append(zscoreMatrix[i][j])
                maxAmpFreqs[k].append(ampFreqs[i])
                maxPhaseFreqs[k].append(phaseFreqs[j])
            

        # Plot boxplots
        if plotBoxplot:
            fontsiz = 24

            figsize = (8, 10)
            measures = ['maxMods', 'maxZscores', 'maxPhaseFreqs', 'maxAmpFreqs']
            titles = {'maxMods': 'Max mod index', 'maxZscores': 'Mod index shuffle z-score', 'maxPhaseFreqs': 'Freq for phase (Hz)', 'maxAmpFreqs': 'Freq for amplitude (Hz)'}
            cols = ['electrode', 'seed'] + measures
            rows = [[key, iseed, maxMods[key][iseed], maxZscores[key][iseed], maxPhaseFreqs[key][iseed], maxAmpFreqs[key][iseed]] for key in electrodes for iseed in range(nseeds)]

            #maxModsProjs = sorted(maxMods, key=lambda k: np.mean(maxMods[k]), reverse=True)[:maxProjs]

            df = pd.DataFrame(rows, columns=cols)
            fig = plt.figure(figsize=figsize)

            # popColors['IT2/3'] = popColors['IT2']
            # popColors['upper PT5B'] = [x/255.0 for x in [87,104,252]]  #popColors['PT5B']
            # popColors['lower PT5B'] = [x/255.0 for x in [42,51,123]] #'darkcyan'
            # my_palette = {k: popColors[k] for k in includeLabels}  # alternative: 'Set3'

            for imeasure, measure in enumerate(measures):
                plt.subplot(2, 2, imeasure+1)
                sb.stripplot(x='electrode', order=electrodes, y=measure, data=df, jitter=True, split=True,linewidth=1,edgecolor='gray')
                ax = sb.boxplot(x='electrode', order=electrodes, y=measure, data=df, showfliers=False)
                handles, labels = ax.get_legend_handles_labels()
                plt.ylabel(titles[measure], fontsize=fontsiz, labelpad=10)#, fontweight='bold')
                if imeasure < len(measures):
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    plt.xlabel('', fontsize=fontsiz)
                else:
                    ax.set_xticklabels(ax.get_xticklabels(),fontsize=fontsiz-2)#, rotation=-75)
                    plt.xlabel('Electrode', fontsize=fontsiz, labelpad=10)#,fontweight='bold')
                    #plt.ylim(20,50)
                #plt.subplots_adjust(bottom=0.2, top=0.92, right=0.99, left=0.1, hspace=0.2)
                #plt.ylim(0, maxVals[stim][measure]*1.25)
                ax = plt.gca()
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                ax.yaxis.grid(True) # Hide the horizontal gridlines
                ax.xaxis.grid(False)  # Show the vertical gridlines
                
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsiz) 
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsiz)
                plt.tight_layout()

            #plt.suptitle('Population rate PSD statistics across N=25 simulations', fontsize=fontsiz+2, fontweight='bold')
            filename = '%s%s_phase_amp_cfc_stats_boxplot_%d_%d_ampStep_%d.png' % (root, batchLabel, timeRange[0], timeRange[1], ampStep)
            plt.savefig(filename, dpi=300)

# ----------------------------------------------------------------
def fig_osc_spikes():
    ''' -Figure "slow wave recruits PT subpops and modulates gamma freq"
         a) Firing rate histogram 
         b) low pass filtered and normalized 
         c) PSD PT up vs down 
         d) NCD PT up vs down 
 '''

    # ---------------------------------------------------------------------------------------------------------------
    # options

    spikeHist = 0
    bistability_signal = 1
    bistability_updown = 0
    plotAll = 0
    
    dataFolder = '../data/'
    batchLabel = 'v53_batch12'
    simLabel = 'v53_batch12_0_0_0'
    timeRange = [1000, 5000]

    sim, data, out, root = loadSimData(dataFolder, batchLabel, simLabel)

    plt.style.use('seaborn-ticks') 


    # ---------------------------------------------------------------------------------------------------------------
    # spike histogram
    if spikeHist or plotAll:
        fontsiz=16
        L5Bmin=0.47
        L5Bmax=0.8
        L5Bmid = L5Bmin + (L5Bmax-L5Bmin)/2
        upperPT5B = tuple([c['gid'] for c in sim.net.allCells if L5Bmin <= c['tags']['ynorm'] <= L5Bmid and c['tags']['pop']=='PT5B'])
        lowerPT5B = tuple([c['gid'] for c in sim.net.allCells if L5Bmid <= c['tags']['ynorm'] <= L5Bmax and c['tags']['pop']=='PT5B'])
        #popColors[upperPT5B] = popColors['PT5B']
        #popColors[lowerPT5B] = 'darkcyan'
        popColors[upperPT5B] = [x/255.0 for x in [87,104,252]]  #popColors['PT5B']
        popColors[lowerPT5B] = [x/255.0 for x in [42,51,123]] #'darkcyan'
        popColors[('IT2','IT5A')] = popColors['IT2']
        #timeRange = [1000, 4000] # change to 51 sec
        include = [('IT2','IT5A'), 'IT5B',  upperPT5B, lowerPT5B]

        filename = root+simLabel+'spikeHist_%d_%d_10.png'%(timeRange[0], timeRange[1])
        
        fig, data = sim.analysis.plotSpikeHist(include=include, timeRange=timeRange, binSize=10, overlay=True, graphType='line', yaxis = 'rate', 
        popColors = popColors, dpi = 200, figSize = (8,4), axis = 'off', saveFig=0, showFig=0)
        histoData, histoT = data['histoData'], data['histoT'] 

        # plot
        plt.figure(figsize=(8,4))
        colors = [popColors[x] for x in include]
        ySpacing = [0,17, 17+28, 17+28+12]
        for i, histoCount in enumerate(histoData):
            histoCount = np.array(histoCount)-ySpacing[i]
            plt.plot(histoT, histoCount, linewidth=1.0, color=colors[i])
        ax = plt.gca()
        plt.setp(ax.lines, linewidth=0.5)
        plt.title('Spike time histogram', fontweight='bold', fontsize=fontsiz)
        
        # scalebar
        #round_to_n = lambda x, n, m: int(np.round(round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1)) / m)) * m 
        #sizex = round_to_n((timeRange[1]-timeRange[0])/10.0, 1, 50)
        sizex=500
        add_scalebar(ax, hidex=False, hidey=True, matchx=False, matchy=True, sizex=sizex, sizey=None, 
                    unitsx='ms', unitsy='spikes/sec', scalex=1, scaley=1, loc=1, pad=-2, borderpad=0.5, sep=4, prop=None, barcolor="black", barwidth=2)   #4
        plt.axis('off')

        ## legend
        #plt.tight_layout()
        # plt.legend(['IT L2/3,5A', 'PT upper L5B', 'PT lower L5B'], fontsize=fontsiz, bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0)
        # plt.subplots_adjust(right=(0.9))
        plt.setp(ax.lines, linewidth=0.5)
        plt.savefig(filename, dpi=300)

    # ---------------------------------------------------------------------------------------------------------------
    # bistability signals
    if bistability_signal or plotAll:

        # LFP low pass filtered 
        electrodes = [6] #range(11) #+avg
        ''' electrode locations = range(200,1300,100)
        0=200 (L23), 1=300 (L23), 2=400 (L23/4 border), 3=500 (L4/5A border), 4=600 (L5A), 5=700 (L5Bu), 
        6=800 (L5Bu), 7=900 (L5Bm), 8=1000 (L5Bl), 9=1100 (L6), 10=1200 (L6), 11=1300 (L6)
        '''

        # get lfp data - 30-50Hz
        
        plots = ['timeSeries']
        filename= '%s%s_lfp_timeSeries_filtered_%d_%d_30_50hz.png' % (root, simLabel, timeRange[0], timeRange[1])
        fig4,data = sim.analysis.plotLFP(plots=plots, electrodes=electrodes, timeRange=timeRange, figSize=(8,4), overlay=True, 
            separation=1.5, dpi=400, filtFreq=[30,50], detrend=0, normSignal=1, colors=[[0,0,0]]*11,  saveFig=filename, showFig=False) 
        lfpSignal = data['LFP'][:, electrodes[0]]

        # get lfp data - 0-2Hz
        #timeRange = [1000, 5000]
        plots = ['timeSeries']
        filename= '%s%s_lfp_timeSeries_filtered_%d_%d.png' % (root, simLabel, timeRange[0], timeRange[1])
        fig4,data = sim.analysis.plotLFP(plots=plots, electrodes=electrodes, timeRange=timeRange, figSize=(8,4), overlay=True, 
            separation=1.5, dpi=400, filtFreq=2, detrend=0, normSignal=1, colors=[[0,0,0]]*11,  saveFig=False, showFig=False) 
        lfpSignal = data['LFP'][:, electrodes[0]]


        # rate envelope -- PT5B upper correlates with IT2/5A; PT5B lower inversely correlates with IT2/5A 
        fontsiz=16
        L5Bmin=0.47
        L5Bmax=0.8
        L5Bmid = L5Bmin + (L5Bmax-L5Bmin)/2
        upperPT5B = tuple([c['gid'] for c in sim.net.allCells if L5Bmin <= c['tags']['ynorm'] <= L5Bmid and c['tags']['pop']=='PT5B'])
        lowerPT5B = tuple([c['gid'] for c in sim.net.allCells if L5Bmid <= c['tags']['ynorm'] <= L5Bmax and c['tags']['pop']=='PT5B'])
        # popColors[upperPT5B] = popColors['PT5B']
        # popColors[lowerPT5B] = 'darkcyan'
        popColors[upperPT5B] = [x/255.0 for x in [87,104,252]]  #popColors['PT5B']
        popColors[lowerPT5B] = [x/255.0 for x in [42,51,123]] #'darkcyan'
        popColors[('IT2','IT5A')] = popColors['IT5A']
        #timeRange = [1000, 5000] # change to 51 sec
        include = [('IT2','IT5A'), 'IT5B',  upperPT5B, lowerPT5B]

        filename = root+batchLabel+'_rateEnvelope_%d_%d.png'%(timeRange[0], timeRange[1])
        fig1, data = sim.analysis.plotSpikeHist(include=include, timeRange=timeRange, binSize=50, overlay=True, graphType='line', yaxis = 'rate', 
        popColors = popColors, dpi = 200, norm=2, figSize = (16,6), axis = 'on', filtFreq=1.5, smooth=None,  saveFig=0, showFig=0)
        histoData, histoT = data['histoData'], data['histoT']
        # ax = plt.gca()
        # plt.setp(ax.lines, linewidth=3)
        # plt.legend(['IT L2/3,5A', 'PT L5B upper', 'PT L5B lower'], fontsize=fontsiz)
        # plt.savefig(filename, dpi=200)

        # combine lfp + spike hist
        filename= '%s%s_lfp_timeSeries_rateEnvelope_combined_%d_%d.png' % (root, simLabel, timeRange[0], timeRange[1])
        plt.figure(figsize=(8,4))        
        t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)
        # plt.plot(histoT, histoData[0], color=popColors['IT5A'], linewidth=3, label='IT L2/3,5A')
        # plt.plot(histoT, histoData[1], color=popColors['PT5B'], linewidth=3, label='PT upper L5B')
        # plt.plot(histoT, histoData[2], color='darkcyan', linewidth=3, label='PT lower L5B')
        plt.plot(histoT, histoData[0], color=popColors['IT2'], linewidth=3, label='IT L2/3,5A')
        #plt.plot(histoT, histoData[1], color=popColors['IT5A'], linewidth=3, label='IT L5A')
        plt.plot(histoT, histoData[1], color=popColors['IT5B'], linewidth=3, label='IT 5B')
        plt.plot(histoT, histoData[2], color=popColors[upperPT5B], linewidth=3, label='PT upper L5B')
        plt.plot(histoT, histoData[3], color=popColors[lowerPT5B], linewidth=3, label='PT lower L5B')
        
        plt.plot(t, lfpSignal, 'k:', linewidth=3, label='LFP')
        plt.axis('off')
        ax=plt.gca()
        plt.title('Spike time histogram and LFP (low pass filtered and normalized)', fontweight='bold', fontsize=fontsiz)
        filename= '%s%s_lfp_timeSeries_rateEnvelope_combined_%d_%d.png' % (root, simLabel, timeRange[0], timeRange[1])
        plt.savefig(filename, dpi=300)

        # Code to plot legend separately
        # filename= '%s%s_lfp_timeSeries_rateEnvelope_combined_%d_%d_legend.png' % (root, simLabel, timeRange[0], timeRange[1])
        # plt.legend(fontsize=fontsiz, loc=1, ncol=3)
        # plt.ylim([0,4.0])
        # plt.savefig(filename, dpi=300)


    # bistability PSD
    if bistability_updown or plotAll:
        # find peaks and troughs
        from peakutils.peak import indexes 
        ptwave = histoData[2]
        peaksUp = indexes(ptwave, thres=0.3, min_dist=600/50.0)
        peaksDown = indexes(-ptwave, thres=0.3, min_dist=600/50.0)
        peaks = np.array(list(peaksUp)+list(peaksDown))
        peaks = np.sort(peaks)

        midpoints = [peaks[i]+int(np.ceil((peaks[i+1]-peaks[i])/2.0)) for i in range(0, len(peaks)-1)]
        plt.figure()
        plt.plot(histoT, ptwave, color=popColors['PT5B'], linewidth=3, label='PT L5B upper') 
        [plt.plot([timeRange[0]+((x)*50)]*2,[0,1]) for x in list(peaks)+list(midpoints)]

        ptwaveUp = np.array([None]*len(ptwave))
        ptwaveDown = np.array([None]*len(ptwave))

        if peaksDown[0] < peaksUp[0]:
            state = 'down'
        else:
            state = 'up'
        current = 0
        for mp in midpoints:
            if state=='down':
                ptwaveDown[current:mp+1] = ptwave[current:mp+1]
                state='up'
            elif state=='up':
                ptwaveUp[current:mp+1] = ptwave[current:mp+1]
                state='down'
            current = mp

        plt.plot(ptwaveUp, color='green', linewidth=2, label='PT L5B UP')
        plt.plot(ptwaveDown, color='red', linewidth=2, label='PT L5B DOWN') 


        ## Compute slow wave up vs down phase time regions (peaks, troughs and midpoints)
        spktOrig, spkidOrig, lfpOrig = np.array(sim.allSimData['spkt']), np.array(sim.allSimData['spkid']), list(sim.allSimData['LFP'])
        
        spktUp, spkidUp, lfpUp = [], [], []
        spktDown, spkidDown, lfpDown = [], [], []
        current = 0
        offset = timeRange[0]
        step = 50
        midpoints.append(timeRange[1])  # last point

        for mp in midpoints:
            if state=='down':
                #ptwaveDown[current:mp+1] = ptwave[current:mp+1]
                spktDown.extend(list(spktOrig[(offset+(current*step)<=spktOrig) & (spktOrig<=offset+(mp+1)*step)]))
                spkidDown.extend(list(spkidOrig[(offset+(current*step)<=spktOrig) & (spktOrig<=offset+(mp+1)*step)]))
                lfpDown.extend(lfpOrig[int((offset+(current*step))/sim.cfg.recordStep):int((offset+((mp+1)*step))/sim.cfg.recordStep)])
                state='up'
            elif state=='up':
                #ptwaveUp[current:mp+1] = ptwave[current:mp+1]
                spktUp.extend(list(spktOrig[(offset+(current*step)<=spktOrig) & (spktOrig<=offset+(mp+1)*step)]))
                spkidUp.extend(list(spkidOrig[(offset+(current*step)<=spktOrig) & (spktOrig<=offset+(mp+1)*step)]))
                lfpUp.extend(lfpOrig[int((offset+(current*step))/sim.cfg.recordStep):int((offset+((mp+1)*step))/sim.cfg.recordStep)])
                state='down'
            current = mp

        #----------------
        # Up vs Down figs
        raster = 0
        ratePSD = 1
        lfpPSD = 0
        rateNCD = 1

        #----------------
        # DOWN state figs
        sim.allSimData['spkt'] = spktDown
        sim.allSimData['spkid'] = spkidDown
        sim.allSimData['LFP'] = lfpDown
        colorUp = [x/255.0 for x in [110,24,255]]
        
        ## raster
        if raster:
            filename = root+batchLabel+'_down_raster_%d_%d.png'%(timeRange[0], timeRange[1])
            sim.analysis.plotRaster(include=['PT5B'], timeRange=timeRange, orderBy=['pop', 'y'], orderInverse=1, popColors=popColors, saveFig=filename)

        # rate PSD
        if ratePSD:
            filename = root+batchLabel+'_down_ratePSD_%d_%d.png'%(timeRange[0], timeRange[1])
            fig, allSignal, allPower, allFreqs = sim.analysis.plotRatePSD(include=['PT5B'], timeRange=timeRange, maxFreq=80, NFFT=128, noverlap=64, smooth=8,popColors=popColors, showFig=0) 
            ratePSDDown = allSignal[0]
            fig, allSignal, allPower, allFreqs = sim.analysis.plotRatePSD(include=[upperPT5B], timeRange=timeRange, maxFreq=80, NFFT=128, noverlap=64, smooth=8,popColors=popColors, showFig=0) 
            ratePSDDown_upper = allSignal[0]


        ## LFP PSD
        if lfpPSD:
            # filename = root+batchLabel+'_down_lfpPSD_%d_%d.png'%(timeRange[0], timeRange[1])
            fig, data = sim.analysis.plotLFP(plots= ['PSD'], electrodes=[6], figSize=(6,6), timeRange=timeRange, NFFT=128*20, noverlap=0.9*128*20, nperseg=128*20, showFig=0)
            lfpPSDDown = data['allSignal'][0]

        ## PT5B rate vs NCD
        if rateNCD:
            #timeRange = [1000, 5000]
            include = ['PT5B']
            bins = 8
            fig1, statData, gidsData, ynormsData = sim.analysis.plotSpikeStats(include=include, graphType='scatter', figSize=(7,6), timeRange=timeRange, 
                stats=['rate'], bins=bins,fontsize=fontsiz, popColors=popColors, dpi=200, showFig=0, saveFig=0)
            rateNCDDown = {'ynormsData': ynormsData, 'statData': statData}


        #--------------
        # UP state figs
        sim.allSimData['spkt'] = spktUp
        sim.allSimData['spkid'] = spkidUp
        sim.allSimData['LFP'] = lfpUp
        colorDown = [x/255.0 for x in [24,185,255]]

        ## raster
        if raster:
            filename = root+simLabel+'_up_raster_%d_%d.png'%(timeRange[0], timeRange[1])
            sim.analysis.plotRaster(include=['PT5B'], timeRange=timeRange, orderBy=['pop', 'y'], orderInverse=1, popColors=popColors, saveFig=filename)

        ## rate PSD and plot combined
        if ratePSD:
            filename = root+simLabel+'_UpDown_ratePSD_%d_%d_both.png'%(timeRange[0], timeRange[1])
            fig, allSignal, allPower, allFreqs = sim.analysis.plotRatePSD(include=['PT5B'], timeRange=timeRange, maxFreq=80, NFFT=128, noverlap=64, smooth=8,popColors=popColors, showFig=0) 
            ratePSDUp = allSignal[0]
            fig, allSignal, allPower, allFreqs = sim.analysis.plotRatePSD(include=[upperPT5B], timeRange=timeRange, maxFreq=80, NFFT=128, noverlap=64, smooth=8,popColors=popColors, showFig=0) 
            ratePSDUp_upper = allSignal[0]
            ratePSDFreqs = allFreqs[0]

            maxFreq = 80
            plt.figure(figsize=(8,4))
            plt.plot(ratePSDFreqs[ratePSDFreqs<maxFreq], ratePSDUp[ratePSDFreqs<maxFreq], linewidth=1.5, color=colorUp, label='PT5B during delta wave crests')
            plt.plot(ratePSDFreqs[ratePSDFreqs<maxFreq], ratePSDDown[ratePSDFreqs<maxFreq], linewidth=1.5, color=colorDown, label='PT5B during delta wave troughs')
            plt.plot(ratePSDFreqs[ratePSDFreqs<maxFreq], ratePSDUp_upper[ratePSDFreqs<maxFreq], linewidth=1.5, ls=':', color=colorUp, label='upper PT5B during delta wave crests')
            plt.plot(ratePSDFreqs[ratePSDFreqs<maxFreq], ratePSDDown_upper[ratePSDFreqs<maxFreq], linewidth=1.5, ls=':', color=colorDown, label='upper PT5B during delta wave troughs')

            plt.xlabel('Frequency (Hz)', fontsize=fontsiz-2)
            plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=fontsiz-2) # add yaxis in opposite side
            plt.xlim([0, maxFreq])
            plt.ylim([-22,20])
            plt.legend(fontsize=fontsiz-3)
            plt.subplots_adjust(bottom=(0.15))
            plt.savefig(filename, dpi=300)

        ## LFP PSD and plot combined
        if lfpPSD:
            filename = root+simLabel+'_upDown_lfpPSD_%d_%d.png'%(timeRange[0], timeRange[1])
            fig, data = sim.analysis.plotLFP(plots= ['PSD'], electrodes=[6], figSize=(6,6), timeRange=timeRange, NFFT=128*20, noverlap=0.9*128*20, nperseg=128*20, showFig=0)
            lfpPSDUp = data['allSignal'][0]
            lfpPSDFreqs = data['allFreqs'][0]

            maxFreq = 100
            plt.figure(figsize=(6,6))
            plt.plot(lfpPSDFreqs[lfpPSDFreqs<maxFreq], lfpPSDUp[lfpPSDFreqs<maxFreq], linewidth=1.5, color=colorUp, label='LFP during delta wave crests')
            plt.plot(lfpPSDFreqs[lfpPSDFreqs<maxFreq], lfpPSDDown[lfpPSDFreqs<maxFreq], linewidth=1.5, color=colorDown, label='LFP during delta wave troughs')
            plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
            plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=fontsiz) # add yaxis in opposite side
            plt.xlim([0, maxFreq])
            plt.legend(fontsize=16)
            plt.savefig(filename, dpi=200)

        ## PT5B rate vs NCD and plot combined
        if rateNCD:
            fontsiz = 16
            #timeRange = [1000, 5000]
            include = ['PT5B']
            filename = root+simLabel+'_%d_%d_exc'%(timeRange[0], timeRange[1])
            fig1, statData, gidsData, ynormsData = sim.analysis.plotSpikeStats(include=include, graphType='scatter', figSize=(7,6),  
                timeRange=timeRange, stats=['rate'], bins=bins,fontsize=fontsiz, popColors=popColors, 
                dpi=200, showFig=0, saveFig=0)
            rateNCDUp = {'ynormsData': ynormsData, 'statData': statData}

            # plot 
            plt.figure(figsize=(8,4))
            ynormsData = [rateNCDUp['ynormsData'][0], rateNCDDown['ynormsData'][0]]
            statData = [rateNCDUp['statData'][0], rateNCDDown['statData'][0]]
            legendLabels = ['PT5B during delta wave crests','PT5B during delta wave troughs']
            colors = [colorUp, colorDown]

            from scipy import stats
            for i,(ynorms,data) in enumerate(zip(ynormsData, statData)):
                mean, binedges, _ = stats.binned_statistic(ynorms, data, 'mean', bins=bins)
                median, binedges, _ = stats.binned_statistic(ynorms, data, 'median', bins=bins)
                std, binedges, _ = stats.binned_statistic(ynorms, data, 'std', bins=bins)
                label = legendLabels[i]
                #plt.scatter(ynorms, data, color=colors[i], label=label, s=4) #[88/255.0,204/255.0,20/255.0]
                binstep = binedges[1]-binedges[0]
                bincenters = [b+binstep/2 for b in binedges[:-1]] 
                #plt.errorbar(bincenters, mean, yerr=std, color=colors[i], fmt = 'o-',capthick=1, capsize=5) #[44/255.0,53/255.0,127/255.0]
                plt.plot(bincenters, mean, 'o-', color=colors[i], label=label) #[44/255.0,53/255.0,127/255.0]

            ax = plt.gca()
            plt.xlabel('Normalized cortical depth (NCD)', fontsize=fontsiz-2)
            plt.ylabel('Avg firing rate (Hz)',fontsize=fontsiz-2)
            plt.subplots_adjust(bottom=(0.15))
            plt.legend(fontsize=fontsiz-3)
            #ax.legend_.remove()
            filename = root+simLabel+'_UpDown_rateNCD _%d_%d.png'%(timeRange[0], timeRange[1])
            plt.ylim([0,10])
            plt.savefig(filename, dpi=300)



# ----------------------------------------------------------------
def fig_lfp_ih():
    ''' Figure comparing LFP spectogram with and w/o ih: 
        - LFP power: x-axis=time; y-axis=electrode; color=power (0-4Hz or 25-40Hz) 
        '''

    # ---------------------------------------------------------------------------------------------------------------
    # options

    lfp_power = 1
    plotAll = 0
   
    dataFolder = '../data/'
    
    # single sim
    batchLabel = 'v53_batch12'
    simLabels = ['v53_batch12_0_0_0', 'v53_batch12_2_0_0']
    ihLabels = ['Low ih', 'High ih']
    fontsiz = 18
    # N=25
    #batchLabel = 'v53_batch13'
    #simLabels =  ['v53_batch13_0_'+str(iseed)+'_'+str(iconn)+'_0' for iseed in range(5) for iconn in range(5)]
        
    import matplotlib.cm as cm
    plt.style.use('seaborn-ticks')
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
            
    for iih, simLabel in enumerate(simLabels):
        sim, data, out, root = loadSimData(dataFolder, batchLabel, simLabel)

        # ---------------------------------------------------------------------------------------------------------------
        # LFP Spectrogram
        if lfp_power or plotAll:
            def ms2index (ms, sampr): return int(sampr*ms/1e3) # millisecond to index; sampr=sampling rate in Hz

            # spectrogram
            timeRange = [1000, 5000]
            electrodes = range(11)
            ''' electrode locations = range(200,1300,100)
            0=200 (L23), 1=300 (L23), 2=400 (L23/4 border), 3=500 (L4/5A border), 4=600 (L5A), 5=700 (L5Bu), 
            6=800 (L5Bu), 7=900 (L5Bm), 8=1000 (L5Bl), 9=1100 (L6), 10=1200 (L6) (11=1300 (L6))
            '''
            elecDepths = range(200,1300,100)
            plots = ['spectrogram']
            maxFreq = [80, 80]
            freqRanges = [[20,30], [30,40]] # plots with err bar / PSD with err bar or overlayed (gray)

            for ifreqRange, freqRange in enumerate(freqRanges):

                tmpfig, data = sim.analysis.plotLFP(plots=plots, electrodes=electrodes, timeRange=timeRange, figSize=(8,20), overlay=True, maxFreq=maxFreq[ifreqRange], normSpec=False, dpi=300, saveFig=0, showFig=False) 

                # data = list[11 electrodes]array[41 freqs, 293 time]
                spec = data['spec']
                freqs = data['freqs']
                t = range(spec[0].TFR.shape[1])

                power = np.zeros((len(electrodes), len(t)))  # num electrodes x time
                for ielec, elec in enumerate(electrodes):
                    power[ielec,:] = np.mean(spec[ielec].TFR[(freqs > freqRange[0]) & (freqs < freqRange[1]), :], 0)

                if ifreqRange==0 and iih==0:
                    vmin = power.min()
                    vmax = power.max()
                
                # plt.imshow(S, extent=(np.amin(T), np.amax(T), np.amin(F), np.amax(F)), origin='lower', interpolation='None', aspect='auto', vmin=vc[0], vmax=vc[1], cmap=plt.get_cmap('viridis'))
                # plt.colorbar(label='Power')
                # plt.ylabel('Hz')
                # plt.tight_layout()   

                # hard code min and max values 0 - 0.0022
                # and renormalize between 0 and 1.0
                vmin = 0.0
                vmax = 0.0022
                power = power/vmax 

                isub = (ifreqRange * 2) + iih + 1
                plt.sca(axes[ifreqRange,iih])
                plt.subplot(2, 2, isub)
                plt.pcolor(power, cmap=cm.viridis, vmin=0.0, vmax=1.0)
                #plt.plot(power)
                #axes[ifreqRange, iih].set_
                #plt.suptitle('%s, %d-%d Hz' % (ihLabels[iih], freqRange[0], freqRange[1]), fontsize=fontsiz,  fontweight='bold')
                plt.ylabel('Electrode depth (um)',fontsize=fontsiz)
                plt.xlabel('Time (s)', fontsize=fontsiz)
                xtickLabels = range(1,6) #[range(timeRange[0]+500, timeRange[1], 1000)
                xtickValues = [ms2index((x*1000)-timeRange[0], 10000) for x in xtickLabels] #tickLabe#np.interp([x-timeRange[0] for x in xtickLabels], t, range(len(t)))
                plt.xticks(xtickValues, xtickLabels)
                plt.yticks([x+0.5 for x in range(len(elecDepths))], elecDepths)
                plt.gca().invert_yaxis()
                plt.setp(axes[ifreqRange,iih].get_xticklabels(),fontsize=fontsiz-2)
                plt.setp(axes[ifreqRange,iih].get_yticklabels(),fontsize=fontsiz-2)
                plt.subplots_adjust(bottom=0.1, top=0.9, right=1.0, left=0.1)
                cbar = plt.colorbar()
                cbar.set_label(label='Normalized average power',fontsize=fontsiz-2)
                plt.tight_layout()


            
            
    plt.savefig('%s%s_lfp_ih_power_%d_%d_freqs_%d_%d.png' % (root, simLabel, timeRange[0], timeRange[1], freqRanges[0][0], freqRanges[1][0]), dpi=300)


# ----------------------------------------------------------------
def fig_psd():
    ''' -Figure PSDs: 
        - LFP PSD
        - Rate PSD 1-6 sec, all, IT2, IT5A, PT5B
        '''

    # ---------------------------------------------------------------------------------------------------------------
    # options

    psd_spikes = 0
    psd_lfp = 0
    psd_spikes_stats = 0
    psd_lfp_stats = 1
    plotAll = 0
   
    dataFolder = '../data/'
    
    # # single sim 5 sec
    # batchLabel = 'v53_batch12'
    # simLabels = ['v53_batch12_0_0_0']
    
    # # single sim 50 sec
    # batchLabel = 'v53_batch11'
    # simLabels = ['v53_batch11_0_0']
    
    
    #N=25
    batchLabel = 'v53_batch13'
    simLabels = ['v53_batch13_0_' + str(iseed) + '_' + str(iconn) + '_0' for iseed in range(5) for iconn in range(5)]

    plt.style.use('seaborn-ticks') 
    
    if psd_spikes or psd_lfp or plotAll:

        root = dataFolder + batchLabel + '/'

        saveToFile = 1
        dataSaveSpikes = []
        dataSaveLFP = []
        
        for simLabel in simLabels:
            sim, data, out, root = loadSimData(dataFolder, batchLabel, simLabel)

            # ---------------------------------------------------------------------------------------------------------------
            # rate psd
            fontsiz=12
            figsiz = (8,4)
            if psd_spikes or plotAll:
                
                timeRange = [1000, 5000]
                L5Bmin=0.47
                L5Bmax=0.8
                L5Bmid = L5Bmin + (L5Bmax-L5Bmin)/2
                upperPT5B = tuple([c['gid'] for c in sim.net.allCells if L5Bmin <= c['tags']['ynorm'] <= L5Bmid and c['tags']['pop']=='PT5B'])
                lowerPT5B = tuple([c['gid'] for c in sim.net.allCells if L5Bmid <= c['tags']['ynorm'] <= L5Bmax and c['tags']['pop']=='PT5B'])
                IT245A = tuple(['IT2', 'IT4', 'IT5A'])
                include = ['IT2', 'IT5A', 'IT5B', upperPT5B, lowerPT5B] #, 'IT6'] #['allCells', 'IT2','IT4','IT5A','IT5B', upperPT5B, lowerPT5B, 'IT6','CT6'] #+ 
                includeLabels = ['IT2/3', 'IT5A', 'IT5B', 'upper PT5B', 'lower PT5B']
                excpops # 'IT2', 'IT5A', 'PT5B']
                popColors['allCells'] = 'k'
                popColors[upperPT5B] = [x/255.0 for x in [87,104,252]]  #popColors['PT5B']
                popColors[lowerPT5B] = [x/255.0 for x in [42,51,123]] #'darkcyan'
                popColors[IT245A] = popColors['IT5A'] 

                fig3, outData = sim.analysis.plotRatePSD(include=include, timeRange=timeRange, maxFreq=80, stepFreq=1.0, NFFT=128, noverlap=64, smooth=8, norm=1, showFig=0, saveFig=0, popColors=popColors, lineWidth=2.5, binSize=4, figSize=figsiz)
                # ylim=[-55, 0]
                ax = plt.gca()
                plt.title('') #Population PSD', fontsize=fontsiz, fontweight='bold')
                plt.xlabel('Frequency (Hz) ', fontsize=fontsiz) #Frequency (Hz)', fontsize=fontsiz)
                plt.ylabel('Normalized population rate power', fontsize=fontsiz)
                ax = plt.gca()
                plt.setp(ax.get_xticklabels(),fontsize=fontsiz-2)
                plt.setp(ax.get_yticklabels(),fontsize=fontsiz-2)
                #plt.ylim(-29,15)
                plt.subplots_adjust(bottom=0.17, top=0.95, right=0.9, left=0.1)
                #plt.tight_layout()
                plt.legend(includeLabels, fontsize=fontsiz-4) #, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0) 
                # ['allCells', 'IT2','IT4','IT5A','IT5B', 'upper PT5B', 'lower PT5B', 'IT6','CT6']
                plt.savefig('%s%s_spike_psd_morlet_%d_%d.png'% (root, simLabel, timeRange[0], timeRange[1]), dpi=300)

                signal = {}
                freq = {}
                for ilabel,label in enumerate(includeLabels):
                    signal[label] = outData['allSignal'][ilabel]
                    freq[label] = outData['allFreqs'][ilabel]

                dataSaveSpikes.append({'signal': signal, 'freq': freq})

                if saveToFile:
                    with open('%s%s_psd_spike_stats_data_%d_%d.pkl' % (root, batchLabel, timeRange[0], timeRange[1]), 'wb') as f:
                        pickle.dump(dataSaveSpikes, f)

            # ---------------------------------------------------------------------------------------------------------------
            # LFP PSD
            if psd_lfp or plotAll:

                # psd
                figSiz = (8,4)
                fontsiz=12
                timeRange = [1000, 5000]
                electrodes = [1,4,6,8] #,10]
                legendLabels = ['Electrode 300um (L2/3)', 'Electrode 600um (L5A)', 'Electrode 800um (upper L5B)', 'Electrode 1000um (lower L5B)'] #, 'Electrode 1100um (L6)']
                plots = ['PSD'] #['PSD','timeSeries', 'spectrogram']
                colors = [[0,0,0]]*11
                
                colors[0] = [x/255.0 for x in [253,116,0]]    # popColors['IT2']
                colors[1] = [x/255.0 for x in [255,225,26]]   # popColors['IT5A']
                colors[2] = [x/255.0 for x in [190, 219, 57]] #'mediumpurple'#popColors['PT5B']
                colors[3] = [x/255.0 for x in [31, 138, 112]] #'purple'

                colors[0] = 'red' #popColors['IT2']
                colors[1] = 'magenta' #popColors['IT5A']
                colors[2] = 'blue' #'mediumpurple'#popColors['PT5B']
                colors[3] = 'green' #'purple'
                #colors[4] = 'firebrick' #popColors['IT6']

                filtFreq = 200
                fig4, outData = sim.analysis.plotLFP(plots=plots, electrodes=electrodes, timeRange=timeRange, figSize=figSiz, overlay=True,  maxFreq=80, stepFreq=1, normSignal=0, normPSD=1, separation=1.5, dpi=200, lineWidth=2.5, colors=colors, saveFig=0, showFig=False) 
                plt.legend(legendLabels, fontsize=fontsiz-3)
                ax = plt.gca()
                plt.title('')
                plt.suptitle('')
                #ax.legend_.remove()
                [plt.setp(l,linewidth=2.5) for l in ax.lines] 
                #plt.xscale('log')
                plt.xlabel('Frequency (Hz)', fontsize=fontsiz) 
                plt.ylabel('Normalized LFP power',fontsize=fontsiz)
                plt.setp(ax.get_xticklabels(),fontsize=fontsiz-2)
                plt.setp(ax.get_yticklabels(),fontsize=fontsiz-2)
                
                #fplt.suptitle('LFP power spectral density', fontsize=fontsiz, fontweight='bold')
                plt.subplots_adjust(bottom=0.17, top=0.95, right=0.9, left=0.1)
                plt.savefig('%s%s_lfp_psd_morlet_notnorm_%d_%d_%d.png' % (root, simLabel, timeRange[0], timeRange[1],filtFreq), dpi=300)


                signal = {}
                freq = {}
                for ilabel,label in enumerate(legendLabels):
                    signal[label] = outData['allSignal'][ilabel]
                    freq[label] = outData['allFreqs'][ilabel]

                dataSaveLFP.append({'signal': signal, 'freq': freq})

                if saveToFile:
                    with open('%s%s_psd_lfp_norm_stats_data_%d_%d.pkl' % (root, batchLabel, timeRange[0], timeRange[1]), 'wb') as f:
                        pickle.dump(dataSaveLFP, f)

    # ---------------------------------------------------------------------------------------------------------------
    # Statistics on psd spikes
    if psd_spikes_stats:

        root = dataFolder + batchLabel + '/'
        fontsiz = 12
        figsiz = (8,4)
        timeRange = [1000, 5000]
        binsiz = 4
        nseeds = 25
        includeLabels = ['IT2/3', 'IT5A', 'IT5B', 'upper PT5B', 'lower PT5B']

        plotOverlaid = False
        plotBoxplot = True

        # load and preprocess data
        with open('%s%s_psd_spike_stats_data_%d_%d.pkl' % (root, batchLabel, timeRange[0], timeRange[1]), 'rb') as f:
            data = pickle.load(f)
        
        allData = {}
        maxPeaks = {}
        maxFreqs = {}

        psdPops = [k for k in data[0]['signal'].keys() if k in includeLabels] 

        minFreqIndex = 0  # 8Hz
        f = data[0]['freq'][psdPops[0]][minFreqIndex:-1]

        for k in psdPops:
            allData[k] = []
            maxPeaks[k] = []
            maxFreqs[k] = []
            
            print('Analyzing %s' % (k))
            for iseed in range(nseeds):
                # group data by projection
                psd = data[iseed]['signal'][k][minFreqIndex:-1]
                if plotOverlaid:  # normalize so comparable
                    psd = psd / np.max(psd)
                allData[k].append(psd)
                maxPeaks[k].append(max(psd))
                maxFreqs[k].append(f[np.argmax(psd)])

            # calculate mean and stderr
            mean = np.mean(allData[k], 0)
            import scipy 
            sem = scipy.stats.sem(np.array(allData[k]))
            iqr = scipy.stats.iqr(np.array(allData[k]),0)
            std = np.std(allData[k], 0)
            
            
            if plotOverlaid:

                # plot psd mean with error bars
                figh = plt.figure(figsize=figsiz)
                plt.plot(f, mean, color='blue', linewidth=1.5, label=k)
                plt.legend([k], fontsize=fontsiz)
                plt.errorbar(f, mean, yerr=iqr, fmt='.', color='blue', ecolor='blue', markersize=1, elinewidth=1, capsize=0)
                
                plt.plot([0,0], [0,0], color='black', linewidth=0.5, label=k)
                for iseed in range(nseeds):
                    g = allData[k][iseed]
                    color = [0.5+(nseeds - iseed) / (nseeds*2.0)] * 3
                    plt.plot(f, g, color=color, linewidth=0.5, label=k)

                #plt.title(, fontsize=fontsiz)
                plt.xlim(0,80)
                ax = plt.gca()
                #plt.grid(False)
                plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
                plt.ylabel('Normalized population rate power', fontsize=fontsiz)
                plt.setp(ax.get_yticklabels(),fontsize=fontsiz)
                plt.setp(ax.get_xticklabels(),fontsize=fontsiz)
                plt.subplots_adjust(bottom=0.17, top=0.95, right=0.9, left=0.1)
                filename = '%s%s_psd_lfp_stats_iqr_%s_%d_%d_binSize-%d.png' % (root, batchLabel, k.replace(' ', '').replace('/', ''), timeRange[0], timeRange[1], binsiz)
                plt.savefig(filename, dpi=300)

        # Plot boxplots
        if plotBoxplot:
            fontsiz = 14
            includeRatio = False

            if includeRatio:
                figsize = (10, 12)
                measures = ['maxPeaks', 'maxFreqs', 'maxFreqsRatio']
                titles = {'maxPeaks': 'Peak Power', 'maxFreqs': 'Frequency of peak power (Hz)', 'maxFreqsRatio': 'Frequency of peak power wrt IT2/3 (Hz)'}
                cols = ['pop', 'seed'] + measures
                rows = [[key, iseed, maxPeaks[key][iseed], maxFreqs[key][iseed], maxFreqs[key][iseed]/maxFreqs['IT2/3'][iseed]] for key in psdPops for iseed in range(nseeds)]
            else:
                figsize = (10, 8)
                measures = ['maxPeaks', 'maxFreqs']
                titles = {'maxPeaks': 'Normalized peak power', 'maxFreqs': 'Frequency of peak power (Hz)'}
                cols = ['pop', 'seed'] + measures
                rows = [[key, iseed, maxPeaks[key][iseed], maxFreqs[key][iseed]] for key in psdPops for iseed in range(nseeds)]

            #maxPeaksProjs = sorted(maxPeaks, key=lambda k: np.mean(maxPeaks[k]), reverse=True)[:maxProjs]

            df = pd.DataFrame(rows, columns=cols)
            fig = plt.figure(figsize=figsize)

            popColors['IT2/3'] = popColors['IT2']
            popColors['upper PT5B'] = [x/255.0 for x in [87,104,252]]  #popColors['PT5B']
            popColors['lower PT5B'] = [x/255.0 for x in [42,51,123]] #'darkcyan'
            my_palette = {k: popColors[k] for k in includeLabels}  # alternative: 'Set3'

            for imeasure, measure in enumerate(measures):
                plt.subplot(len(measures), 1, imeasure+1)
                sb.stripplot(x='pop', order=psdPops, y=measure, data=df, palette=my_palette, jitter=True, split=True,linewidth=1,edgecolor='gray')
                ax = sb.boxplot(x='pop', order=psdPops, y=measure, data=df, palette=my_palette, showfliers=False)
                handles, labels = ax.get_legend_handles_labels()
                plt.ylabel(titles[measure], fontsize=fontsiz, labelpad=10, fontweight='bold')
                if imeasure < len(measures)-1:
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    plt.xlabel('', fontsize=fontsiz)
                else:
                    ax.set_xticklabels(ax.get_xticklabels(),fontsize=fontsiz-2)#, rotation=-75)
                    plt.xlabel('Population', fontsize=fontsiz, labelpad=10,fontweight='bold')
                    #plt.ylim(20,50)
                plt.subplots_adjust(bottom=0.2, top=0.92, right=0.99, left=0.1, hspace=0.2)
                #plt.ylim(0, maxVals[stim][measure]*1.25)
                ax = plt.gca()
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                ax.yaxis.grid(True) # Hide the horizontal gridlines
                ax.xaxis.grid(False)  # Show the vertical gridlines
                
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsiz) 
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsiz)
                #plt.tight_layout()

            #plt.suptitle('Population rate PSD statistics across N=25 simulations', fontsize=fontsiz+2, fontweight='bold')
            filename = '%s%s_psd_spikes_stats_boxplot_%d_%d_binSize-%d.png' % (root, batchLabel, timeRange[0], timeRange[1], binsiz)
            plt.savefig(filename, dpi=300)




    # ---------------------------------------------------------------------------------------------------------------
    # Statistics on psd lfp
    if psd_lfp_stats:

        root = dataFolder + batchLabel + '/'
        fontsiz = 12
        figsiz = (8,4)
        timeRange = [1000, 5000]
        binsiz = 4
        nseeds = 25
        includeLabels = ['Electrode 300um (L2/3)', 'Electrode 600um (L5A)', 'Electrode 800um (upper L5B)', 'Electrode 1000um (lower L5B)']  #, 'Electrode 1100um (L6)']
        
        plotOverlaid = 0
        plotBoxplot = 1

        # load and preprocess data
        with open('%s%s_psd_lfp_norm_stats_data_%d_%d.pkl' % (root, batchLabel, timeRange[0], timeRange[1]), 'rb') as f:
            data = pickle.load(f)
        
        allData = {}
        maxPeaks = {}
        maxFreqs = {}

        lfpDepths = [k for k in data[0]['signal'].keys() if k in includeLabels] 

        if plotOverlaid:
            minFreqIndex = 5
            minFreqIndexNorm = 5
        else:
            minFreqIndex = 5  # 5Hz
        maxFreqIndex = 79 # 80hz
        f = data[0]['freq'][lfpDepths[0]][minFreqIndex:maxFreqIndex]

        for k in lfpDepths:
            allData[k] = []
            maxPeaks[k] = []
            maxFreqs[k] = []
            
            print('Analyzing %s' % (k))
            for iseed in range(nseeds):
                # group data by projection
                lfp = data[iseed]['signal'][k][minFreqIndex:maxFreqIndex]
                if plotOverlaid:
                    lfp = lfp / np.max(lfp[minFreqIndexNorm:])
                allData[k].append(lfp)
                maxPeaks[k].append(max(lfp))
                maxFreqs[k].append(f[np.argmax(lfp)])

            # calculate mean and stderr
            mean = np.mean(allData[k], 0)
            import scipy 
            sem = scipy.stats.sem(np.array(allData[k]))
            iqr = scipy.stats.iqr(np.array(allData[k]),0)
            std = np.std(allData[k], 0)
            

            if plotOverlaid:
                # plot psd mean with error bars
                figh = plt.figure(figsize=figsiz)
                plt.plot(f, mean, color='blue', linewidth=1.5, label=k)
                plt.legend([k], fontsize=fontsiz)
                plt.errorbar(f, mean, yerr=iqr, fmt='.', color='blue', ecolor='blue', markersize=1, elinewidth=1, capsize=0)
                
                plt.plot([0,0], [0,0], color='black', linewidth=0.5, label=k)
                for iseed in range(nseeds):
                    g = allData[k][iseed]
                    color = [0.5+(nseeds - iseed) / (nseeds*2.0)] * 3
                    plt.plot(f, g, color=color, linewidth=0.5, label=k)

                #plt.title(, fontsize=fontsiz)
                plt.xlim(0,80)
                ax = plt.gca()
                #plt.grid(False)
                plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
                plt.ylabel('Normalized LFP power', fontsize=fontsiz)
                plt.setp(ax.get_yticklabels(),fontsize=fontsiz)
                plt.setp(ax.get_xticklabels(),fontsize=fontsiz)
                plt.subplots_adjust(bottom=0.17, top=0.95, right=0.9, left=0.1)
                filename = '%s%s_psd_lfp_morlet_notnorm_stats_iqr_%s_%d_%d_binSize-%d.png' % (root, batchLabel, k.replace(' ', '').replace('/', ''), timeRange[0], timeRange[1], binsiz)
                plt.savefig(filename, dpi=300)


        # Plot boxplots
        if plotBoxplot:
            includeRatio = False
            fontsiz = 14
            if includeRatio:
                figsize = (10, 12)
                measures = ['maxPeaks', 'maxFreqs', 'maxFreqsRatio']
                titles = {'maxPeaks': 'Peak power (db/Hz)', 'maxFreqs': 'Frequency of peak power (Hz)', 'maxFreqsRatio': 'Frequency of peak power wrt L2/3 (Hz)'}
                cols = ['depth', 'seed'] + measures
                rows = [[key.replace('Electrode ', ''), iseed, maxPeaks[key][iseed], maxFreqs[key][iseed], maxFreqs[key][iseed]/maxFreqs['Electrode 300um (L2/3)'][iseed]] for key in lfpDepths for iseed in range(nseeds)]
            else:
                figsize = (10, 8)
                measures = ['maxPeaks', 'maxFreqs']
                titles = {'maxPeaks': 'Normalized peak power', 'maxFreqs': 'Frequency of peak power (Hz)'}
                cols = ['depth', 'seed'] + measures
                rows = [[key.replace('Electrode ', ''), iseed, maxPeaks[key][iseed], maxFreqs[key][iseed]] for key in lfpDepths for iseed in range(nseeds)]
           
            #maxPeaksProjs = sorted(maxPeaks, key=lambda k: np.mean(maxPeaks[k]), reverse=True)[:maxProjs]

            df = pd.DataFrame(rows, columns=cols)
            fig = plt.figure(figsize=figsize)

            colors = ['red', 'magenta', 'blue', 'green']
            my_palette = {k.replace('Electrode ', ''): colors[i] for i,k in enumerate(includeLabels)}  # alternative: 'Set3'

            lfpDepthsShort = [k.replace('Electrode ', '') for k in lfpDepths]

            for imeasure, measure in enumerate(measures):
                plt.subplot(len(measures), 1, imeasure + 1)
                sb.stripplot(x='depth', order=lfpDepthsShort, y=measure, data=df, palette=my_palette, jitter=True, split=True,linewidth=1,edgecolor='gray')
                ax = sb.boxplot(x='depth', order=lfpDepthsShort, y=measure, data=df, palette=my_palette, showfliers=False)
                handles, labels = ax.get_legend_handles_labels()
                plt.ylabel(titles[measure], fontsize=fontsiz, labelpad=10, fontweight='bold')
                if imeasure < len(measures)-1:
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    plt.xlabel('', fontsize=fontsiz)
                    #if imeasure == 0: plt.ylim(0.,0.002)
                else:
                    ax.set_xticklabels(ax.get_xticklabels(),fontsize=fontsiz-2)#, rotation=-75)
                    plt.xlabel('LFP electrode depth', fontsize=fontsiz, labelpad=10,fontweight='bold')
                    #plt.ylim(20,50)
                plt.subplots_adjust(bottom=0.2, top=0.92, right=0.99, left=0.13, hspace=0.2)
                ax.yaxis.grid(True) # Hide the horizontal gridlines
                ax.xaxis.grid(False) # Show the vertical gridlines
                #plt.ylim(0, maxVals[stim][measure]*1.25)
                ax = plt.gca()
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsiz) 
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsiz)
                #plt.tight_layout()

            #plt.suptitle('Population rate PSD statistics across N=25 simulations', fontsize=fontsiz+2, fontweight='bold')
            filename = '%s%s_psd_lfp_morlet_notnorm_stats_boxplot_%d_%d_binSize-%d.png' % (root, batchLabel, timeRange[0], timeRange[1], binsiz)
            plt.savefig(filename, dpi=300)

# ----------------------------------------------------------------
def fig_granger():
    ''' -Figure Granger: 
        - Granger 1-6 sec
        - Granger matrix
        - Granger stats / boxplot
        '''

    # ---------------------------------------------------------------------------------------------------------------
    # options

    granger = 0
    granger_matrix = 1
    granger_stats = 0
    plotAll = 0
   
    dataFolder = '../data/'
    
    # single sim 5 sec
    # batchLabel = 'v53_batch12'
    # simLabels = ['v53_batch12_0_0_0']
    
    # # single sim 50 sec
    # batchLabel = 'v53_batch11'
    # simLabels = ['v53_batch11_0_0']
    
    
    #N=25
    batchLabel = 'v53_batch13'
    simLabels = ['v53_batch13_0_' + str(iseed) + '_' + str(iconn) + '_0' for iseed in range(1) for iconn in range(1)]
    
    loadFromFile = 1
    plt.style.use('seaborn-ticks') 
    
    if granger or plotAll:
    
        root = dataFolder + batchLabel + '/'
        
        saveToFile = 0
        dataSave = []
        
        for simLabel in simLabels:
            if not loadFromFile:
                sim, data, out, root = loadSimData(dataFolder, batchLabel, simLabel)

            fontsiz=16
            figsiz = (8,5)

            # ---------------------------------------------------------------------------------------------------------------
            # granger
            if granger or plotAll:

                fontsiz = 12
                figsiz = (8,3)
                timeRange = [1000, 5000]
                grangerPops = ['IT2','SOM2','PV2','IT4','IT5A','SOM5A','PV5A','IT5B','PT5B', 'PT5Bu','PT5Bl','SOM5B','PV5B', 'SOM5','PV5', 'IT6','CT6','SOM6','PV6']
                grangerInclude = {'upperIT_PT': 1, 'PTu_PTl': 1, 'L5_INT': 1, 'IT6_CT6': 1}

                # load output data from file
                if loadFromFile:
                    with open('%s%s_granger_matrix_data_%d_%d.pkl' % (root, batchLabel, timeRange[0], timeRange[1]), 'rb') as f:
                        grangerData = pickle.load(f)[0]
                    binsiz=4

                # compute output data
                else:                            
                    print('Calculating and plotting Granger...')
                    sim.allSimData = data['simData']
                    spkts = sim.allSimData['spkt']
                    spkids = sim.allSimData['spkid']
                    spkids,spkts = zip(*[(spkid,spkt-timeRange[0]) for spkid,spkt in zip(spkids,spkts) 
                        if timeRange[0] <= spkt <= timeRange[1]])

                    L5Bmin=0.47
                    L5Bmax=0.8
                    L5Bmid = L5Bmin + (L5Bmax-L5Bmin)/2
                    upperPT5B = tuple([c['gid'] for c in sim.net.allCells if L5Bmin <= c['tags']['ynorm'] <= L5Bmid and c['tags']['pop']=='PT5B'])
                    lowerPT5B = tuple([c['gid'] for c in sim.net.allCells if L5Bmid <= c['tags']['ynorm'] <= L5Bmax and c['tags']['pop']=='PT5B'])

                    popLabels = sim.net.allPops.keys()
                    gidPops = [cell['tags']['pop'] for cell in sim.net.allCells]
                    popNumCells = [float(gidPops.count(pop)) for pop in popLabels]
                    spktPop = {}
                    for pop, popNum in zip(popLabels, popNumCells):
                        spktPop[pop] = [spkt for spkt,spkid in zip(spkts,spkids) 
                        if sim.net.allCells[int(spkid)]['tags']['pop']==pop]

                    spktPop['PV5'] = spktPop['PV5A']+spktPop['PV5B']
                    spktPop['SOM5'] = spktPop['SOM5A']+spktPop['SOM5B']

                    spktPop['PT5Bu'] = [spkt for spkt,spkid in zip(spkts,spkids) if spkid in upperPT5B]
                    spktPop['PT5Bl'] = [spkt for spkt, spkid in zip(spkts, spkids) if spkid in lowerPT5B]


                    # iterate binsizes and projections
                    binSizes = [4]  # [1,2,3,4,5] -- 1 and 2 not long enough to capture effects; > 2 stable results
                    grangerData = {}
                    for binsiz in binSizes:
                        # compute granger values and z scores for all projections
                        for ipre, prePop in enumerate(grangerPops):
                            for ipost,postPop in enumerate([p for p in grangerPops if p != prePop]):
                                
                                prePopLabel = prePop.replace('IT2', 'IT2/3').replace('PT5Bu', 'upper PT5B').replace('PT5Bl', 'lower PT5B')
                                postPopLabel = postPop.replace('IT2', 'IT2/3').replace('PT5Bu', 'upper PT5B').replace('PT5Bl', 'lower PT5B')
                                y2xProj = ('%s -> %s'%(prePopLabel, postPopLabel))

                                print('Calculating %s ...' % (y2xProj))

                                fig, out = sim.analysis.granger(spks1=spktPop[prePop], spks2=spktPop[postPop], 
                                    label1=prePopLabel, label2=postPopLabel, binSize=binsiz, testGranger=True, plotFig=False)
                                grangerData[y2xProj+'_G'] = out['Fy2x']
                                grangerData[y2xProj+'_Z'] = out['MaxFy2xZscore']

                        grangerData['F'] = out['F']
                    
                    # plot figs for specific groups of proejctions
                    
                if grangerInclude['upperIT_PT']:
                    
                    #IPython.embed()

                    # set line formats
                    lc = {}  # line colors
                    lc['IT4 -> IT2/3'] = 'g-'
                    lc['IT2/3 -> IT4'] = 'g:'
                    lc['IT2/3 -> IT5A'] = 'r-'
                    lc['IT5A -> IT2/3'] = 'r:'
                    lc['IT2/3 -> PT5B'] = 'b-'
                    lc['PT5B -> IT2/3'] = 'b:'
                    lc['IT5A -> PT5B'] = 'm-'
                    lc['PT5B -> IT5A'] = 'm:'

                    # plot

                    figh = plt.figure(figsize=figsiz)
                    for k, v in lc.items():
                        plt.plot(grangerData['F'][1:], grangerData[k+'_G'][1:], v, linewidth=2.5, label=k)

                    plt.legend(fontsize=fontsiz)
                    #plt.ylim(0,4.5)
                    plt.xlim(0,80)
                    ax = plt.gca()
                    plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
                    plt.ylabel('Granger Causality', fontsize=fontsiz)
                    plt.setp(ax.get_xticklabels(),fontsize=fontsiz)
                    plt.setp(ax.get_yticklabels(),fontsize=fontsiz)
                    #plt.tight_layout()
                    #plt.title('Spectral Granger Causality', fontsize=fontsiz, fontweight='bold')
                    plt.subplots_adjust(bottom=0.17, top=0.95, right=0.9, left=0.1)
                    filename='%s%s_granger_upperIT_PT_%d_%d_binSize-%d.png'%(root, simLabel, timeRange[0], timeRange[1], binsiz)
                    plt.savefig(filename, dpi=300)
                    

                # upper vs lower PT
                if grangerInclude['PTu_PTl']:
                    
                    # set line formats
                    lc = {}  # line colors
                    lc['IT2/3 -> upper PT5B'] = 'b-'
                    lc['IT2/3 -> lower PT5B'] = 'r-'
                    lc['IT5A -> upper PT5B'] = 'm-'
                    lc['IT5A -> lower PT5B'] = 'g-'

                    # plot
                    figh = plt.figure(figsize=figsiz)
                    for k, v in lc.items():
                        plt.plot(grangerData['F'][1:], grangerData[k+'_G'][1:], v, linewidth=2.5, label=k)

                    plt.legend(fontsize=fontsiz)
                    #plt.ylim(0,4.5)
                    plt.xlim(0,80)
                    ax = plt.gca()
                    plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
                    plt.ylabel('Granger Causality', fontsize=fontsiz)
                    plt.setp(ax.get_xticklabels(),fontsize=fontsiz)
                    plt.setp(ax.get_yticklabels(),fontsize=fontsiz)
                    #plt.tight_layout()
                    #plt.title('Spectral Granger Causality', fontsize=fontsiz, fontweight='bold')
                    plt.subplots_adjust(bottom=0.17, top=0.95, right=0.9, left=0.1)
                    filename='%s%s_granger_PTu_PTl_%d_%d_binSize-%d.png'%(root, simLabel, timeRange[0], timeRange[1], binsiz)
                    plt.savefig(filename, dpi=300)

                # Interneurons
                if grangerInclude['L5_INT']:
                    lc = {}  # line colors
                    lc['PV5 -> PT5B'] = 'g-'
                    lc['PT5B -> PV5'] = 'g:'
                    lc['PV5 -> IT5A'] = 'r-'
                    lc['IT5A -> PV5'] = 'r:'
                    lc['SOM5 -> PT5B'] = 'b-'
                    lc['PT5B -> SOM5'] = 'b:'
                    lc['SOM5 -> IT5A'] = 'm-'
                    lc['IT5A -> SOM5'] = 'm:'

                    # plot
                    figh = plt.figure(figsize=figsiz)
                    for k, v in lc.items():
                        plt.plot(grangerData['F'][1:], grangerData[k+'_G'][1:], v, linewidth=2.5, label=k)

                    plt.legend(fontsize=fontsiz)
                    #plt.ylim(0,4.5)
                    plt.xlim(0,80)
                    ax = plt.gca()
                    plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
                    plt.ylabel('Granger Causality', fontsize=fontsiz)
                    plt.setp(ax.get_xticklabels(),fontsize=fontsiz)
                    plt.setp(ax.get_yticklabels(),fontsize=fontsiz)
                    #plt.tight_layout()
                    #plt.title('Spectral Granger Causality', fontsize=fontsiz, fontweight='bold')
                    plt.subplots_adjust(bottom=0.17, top=0.95, right=0.9, left=0.1)
                    filename='%s%s_granger_L5_INT_%d_%d_binSize-%d.png'%(root, simLabel, timeRange[0], timeRange[1], binsiz)
                    plt.savefig(filename, dpi=300)


                # L6
                if grangerInclude['IT6_CT6']:
                    lc = {}  # line colors
                    lc['IT6 -> PT5B'] = 'g-'
                    lc['PT5B -> IT6'] = 'g:'
                    lc['IT6 -> IT5A'] = 'r-'
                    lc['IT5A -> IT6'] = 'r:'
                    lc['CT6 -> PT5B'] = 'b-'
                    lc['PT5B -> CT6'] = 'b:'
                    lc['CT6 -> IT5A'] = 'm-'
                    lc['IT5A -> CT6'] = 'm:'

                    # plot
                    figh = plt.figure(figsize=figsiz)
                    for k, v in lc.items():
                        plt.plot(grangerData['F'][1:], grangerData[k+'_G'][1:], v, linewidth=2.5, label=k)

                    plt.legend(fontsize=fontsiz)
                    #plt.ylim(0,4.5)
                    plt.xlim(0,80)
                    ax = plt.gca()
                    plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
                    plt.ylabel('Granger Causality', fontsize=fontsiz)
                    plt.setp(ax.get_xticklabels(),fontsize=fontsiz)
                    plt.setp(ax.get_yticklabels(),fontsize=fontsiz)
                    #plt.tight_layout()
                    #plt.title('Spectral Granger Causality', fontsize=fontsiz, fontweight='bold')
                    plt.subplots_adjust(bottom=0.17, top=0.95, right=0.9, left=0.1)
                    filename='%s%s_granger_IT6_CT6_%d_%d_binSize-%d.png'%(root, simLabel, timeRange[0], timeRange[1], binsiz)
                    plt.savefig(filename, dpi=300)

                dataSave.append(grangerData)

                if saveToFile:
                    with open('%s%s_granger_matrix_data_%d_%d.pkl' % (root, batchLabel, timeRange[0], timeRange[1]), 'wb') as f:
                        pickle.dump(dataSave, f)

    # ---------------------------------------------------------------------------------------------------------------
    # Granger matrix 
    elif granger_matrix:

        root = dataFolder + batchLabel + '/'
        fontsiz = 18
        figsiz = (12,12)
        timeRange = [1000, 5000]
        binsiz = 4
        iseed = -25
        nseeds = 25

        # set font size
        plt.rcParams.update({'font.size': fontsiz})

        matrixPops = ['IT2/3','SOM2','PV2','IT4','IT5A','SOM5A','PV5A','IT5B','upper PT5B','lower PT5B', 'SOM5B','PV5B', 'IT6','CT6','SOM6','PV6']

        # Create plot
        fig = plt.figure(figsize=figsiz)
        h = plt.axes()

        grangerProjs = ['%s -> %s'%(pre,post) for pre in matrixPops for post in matrixPops if pre != post ]

        # load and preprocess data
        with open('%s%s_granger_matrix_data_%d_%d.pkl' % (root, batchLabel, timeRange[0], timeRange[1]), 'rb') as f:
            data = pickle.load(f)

        allData = {}
        maxPeaks = {}
        maxFreqs = {}
        zScores = {}
        for k in grangerProjs:
            allData[k] = []
            maxPeaks[k] = []
            maxFreqs[k] = []
            zScores[k] = []
            print('Analyzing %s'%(k))
            for iseed in range(nseeds):
                # group data by projection
                g = data[iseed][k+'_G'][1:]
                maxPeaks[k].append(max(g))
                maxFreqs[k].append(np.argmax(g)+1)
                zScores[k].append(data[iseed][k + '_Z'])

            # g = data[iseed][k+'_G'][1:]
            # maxPeaks[k] = max(g)
            # maxFreqs[k] = (np.argmax(g))
            # zScores[k] = (data[iseed][k + '_Z'])
                
        #gMatrix = [[maxPeaks['%s -> %s' % (pre, post)] if pre != post else 0] for pre in matrixPops] for post in matrixPops]

        gMatrix = np.zeros((len(matrixPops), len(matrixPops)))
        for ipre, pre in enumerate(matrixPops):
            for ipost, post in enumerate(matrixPops):
                gMatrix[ipost][ipre] = np.mean(maxPeaks['%s -> %s' % (pre, post)]) if pre != post else 0

        plt.imshow(gMatrix, interpolation='nearest', cmap='viridis', vmin=np.nanmin(gMatrix), vmax=np.nanmax(gMatrix))  #_bicolormap(gap=0)

        # Plot grid lines
        matrixPopsNew = matrixPops
        matrixPopsNew[8] = 'PT5Bup'
        matrixPopsNew[9] = 'PT5Blo'

        popsPre, popsPost = matrixPops, matrixPops

        plt.grid(False)
        for ipop, pop in enumerate(popsPre):
            plt.plot(np.array([0,len(popsPre)])-0.5, np.array([ipop,ipop])-0.5, '-', c=(0.7,0.7,0.7))
        for ipop, pop in enumerate(popsPost):
            plt.plot(np.array([ipop,ipop])-0.5, np.array([0,len(popsPost)])-0.5, '-', c=(0.7,0.7,0.7))

        # Make pretty
        h.set_xticks(list(range(len(popsPost))))
        h.set_yticks(list(range(len(popsPre))))
        h.set_xticklabels(popsPost, fontsize=fontsiz-5, rotation=30)
        h.set_yticklabels(popsPre, fontsize=fontsiz-5)
        h.xaxis.set_ticks_position('top')
        plt.xlim(-0.5,len(popsPost)-0.5)
        plt.ylim(len(popsPre) - 0.5, -0.5)

        #if not clim: clim = [np.nanmin(connMatrix), np.nanmax(connMatrix)]
        #plt.clim(clim[0], clim[1])
        plt.colorbar(label='Mean peak Granger causality (N=25)', shrink=0.6) #.set_label(label='Fitness',size=20,weight='bold')
        plt.ylabel('Target population')
        plt.xlabel('Source population')
        h.xaxis.set_label_coords(0.5, 1.11)
        h.yaxis.set_label_coords(-0.12, 0.5)

        #plt.title ('Mean peak Granger Causality matrix', y=1.11)
        
        fig.subplots_adjust(right=0.99) # Less space on right
        fig.subplots_adjust(top=0.98) # Less space on top
        fig.subplots_adjust(bottom=0.01) # Less space on bottom

        filename = '%s%s_granger_matrix_%d_%d_binSize-%d_seed_%d.png' % (root, batchLabel, timeRange[0], timeRange[1], binsiz, iseed)
        plt.savefig(filename, dpi=300)

    # ---------------------------------------------------------------------------------------------------------------
    # Statistics on granger 
    elif granger_stats:

        root = dataFolder + batchLabel + '/'
        fontsiz = 12
        figsiz = (8,3)
        timeRange = [1000, 5000]
        binsiz = 4
        nseeds = 25
        maxProjs = 15

        # load and preprocess data
        with open('%s%s_granger_matrix_data_%d_%d.pkl' % (root, batchLabel, timeRange[0], timeRange[1]), 'rb') as f:
            data = pickle.load(f)
        
        allData = {}
        maxPeaks = {}
        maxFreqs = {}
        zScores = {}
        stderr = {}

        grangerProjs = [k[:-2] for k in data[0].keys() 
            if k.endswith('_G') 
            and 'PV5 ->' not in k
            and not k.endswith('PV5_G')
            and 'SOM5 ->' not in k
            and not k.endswith('SOM5_G')
            and '-> PT5B' not in k
            and not k.startswith('PT5B ->')] # remove PV5, SOM5, and PT5B projections (keep PV5A, PV5B, SOM5A, SOM5B, upper/lower PT5B)

        minFreq = 1
        f = data[0]['F'][minFreq:]

        for k in grangerProjs:
            allData[k] = []
            maxPeaks[k] = []
            maxFreqs[k] = []
            zScores[k] = []
            
            print('Analyzing %s' % (k))
            for iseed in range(nseeds):
                # group data by projection
                g = data[iseed][k+'_G'][minFreq:]
                allData[k].append(g)
                maxPeaks[k].append(max(g))
                maxFreqs[k].append(np.argmax(g)+minFreq)
                zScores[k].append(data[iseed][k+'_Z'])

            plotOverlayed = False
            if plotOverlayed: 
                # calculate mean and stderr
                mean = np.mean(allData[k], 0)
                import scipy 
                sem = scipy.stats.sem(np.array(allData[k]))
                iqr = scipy.stats.iqr(np.array(allData[k]),0)
                std = np.std(allData[k], 0)

                # plot granger mean with error bars
                figh = plt.figure(figsize=figsiz)
                plt.plot(f, mean, color='blue', linewidth=1.5, label=k)
                plt.legend([k], fontsize=fontsiz)
                plt.errorbar(f, mean, yerr=iqr, fmt='.', color='blue', ecolor='blue', markersize=1, elinewidth=1, capsize=0)
                
                plt.plot([0,0], [0,0], color='black', linewidth=0.5, label=k)
                for iseed in range(nseeds):
                    g = data[iseed][k+'_G'][minFreq:]
                    color = [0.5+(nseeds - iseed) / (nseeds*2.0)] * 3
                    plt.plot(f, g, color=color, linewidth=0.5, label=k)

                #plt.title(, fontsize=fontsiz)
                plt.xlim(0,80)
                ax = plt.gca()
                #plt.grid(False)
                plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
                plt.ylabel('Granger Causality', fontsize=fontsiz)
                plt.setp(ax.get_yticklabels(),fontsize=fontsiz)
                plt.setp(ax.get_xticklabels(),fontsize=fontsiz)
                plt.subplots_adjust(bottom=0.17, top=0.95, right=0.9, left=0.1)
                filename = '%s%s_granger_stats_iqr_v2_%s_%d_%d_binSize-%d.png' % (root, batchLabel, k.replace(' ', '').replace('/', ''), timeRange[0], timeRange[1], binsiz)
                plt.savefig(filename, dpi=300)


        # Plot boxplots
        plotBoxplot = True
        if plotBoxplot:
            fontsiz = 18
            figsize = (16, 22)
            measures = ['maxPeaks', 'zScores', 'maxFreqs']
            titles = {'maxPeaks': 'Peak Granger Causality', 'zScores': 'Shuffle test Z-score (significance)', 'maxFreqs': 'Frequency of peak Granger causality (Hz)'}
            #maxVals = {'maxPeaks': }
            cols = ['proj', 'seed'] + measures
            rows = [[key, iseed, maxPeaks[key][iseed], zScores[key][iseed], maxFreqs[key][iseed]] for key in grangerProjs for iseed in range(nseeds)]

            maxPeaksProjs = sorted(maxPeaks, key=lambda k: np.mean(maxPeaks[k]), reverse=True)[:maxProjs]

            df = pd.DataFrame(rows, columns=cols)
            fig = plt.figure(figsize=figsize)

            for imeasure, measure in enumerate(measures):
                plt.subplot(3, 1, imeasure+1)
                sb.stripplot(x='proj', order=maxPeaksProjs, y=measure, data=df, palette='Set3', jitter=True, split=True,linewidth=1,edgecolor='gray')
                ax = sb.boxplot(x='proj', order=maxPeaksProjs, y=measure, data=df, palette='Set3', showfliers=False)
                handles, labels = ax.get_legend_handles_labels()
                plt.tight_layout()
                plt.ylabel(titles[measure], fontsize=fontsiz, fontweight='bold', labelpad=6)
                if imeasure == 0:
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    plt.xlabel('', fontsize=fontsiz)
                elif imeasure == 1:
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    plt.xlabel('', fontsize=fontsiz)
                    plt.ylim(0,500)
                else:
                    ax.set_xticklabels(ax.get_xticklabels(),fontsize=fontsiz-2, rotation=-75)
                    plt.xlabel('Projection', fontsize=fontsiz, fontweight='bold')
                    plt.ylim(20, 50)
                ax.yaxis.grid(True) # Hide the horizontal gridlines
                ax.xaxis.grid(False) # Show the vertical gridlines
                plt.subplots_adjust(bottom=0.17, top=0.97, right=0.99, left=0.08, hspace=0.15)
                #plt.ylim(0, maxVals[stim][measure]*1.25)
                ax = plt.gca()
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsiz) 
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsiz) 

            plt.suptitle('Granger Causality statistics across N=25 simulations', fontsize=fontsiz+2, fontweight='bold')
            filename = '%s%s_granger_stats_boxplot_v2_%d_%d_binSize-%d.png' % (root, batchLabel, timeRange[0], timeRange[1], binsiz)
            plt.savefig(filename, dpi=300)



# ----------------------------------------------------------------
def fig_pulse(batchLabel, simLabels, stim = '', timeRange= [600,1400], histMax = 20):
    ''' Figure pulses: 
        - raster plot of 0.6-1.4 sec 
        - spikeHist
        - rate inc
        - seed rate stats
    '''

    # options
    dataFolder = '../data/'
    loadAll = 0

    allpops = ['IT2','PV2','SOM2','IT4','IT5A','PV5A','SOM5A','IT5B','PT5B','PV5B','SOM5B','IT6','CT6','PV6','SOM6']
    excpops = ['IT2','IT4','IT5A','IT5B','PT5B','IT6','CT6']
    inhpops = ['SOM2','PV2', 'SOM5A','PV5A',  'SOM5B','PV5B',  'SOM6', 'PV6']
    excpopsu = ['IT2','IT4','IT5A','PT5B']

    with open('../sim/cells/popColors.pkl', 'rb') as fileObj: popColors = pickle.load(fileObj)['popColors']
    popColors['S2'] = [0.90,0.76,0.00]
    popColors['M2'] = [0.42,0.67,0.9]
    popColors['TPO'] = [52/255.0, 138/255.0, 49/255.0]

    plt.style.use('seaborn-ticks') 

    # load data
    root = dataFolder+batchLabel+'/'
    
    raster = 0
    spikeHist = 1
    stats = 0
    rates = 0
    seedRates = 0
    calculateRatesData = 0
    plotAll = 0

    loadPickle = 0

    if raster or spikeHist or rates or plotAll:
        for simLabel in simLabels:
            if loadPickle:
                filename = root+simLabel+'.pkl'
                with open(filename[:-4]+'.pkl', 'w') as f:
                    [sim.allSimData, sim.net.allCells, data,out] = pickle.load(f)
            else:
                filename = root+simLabel+'.json'
                sim,data,out = utils.plotsFromFile(filename, raster=0, stats=0, rates=0, syncs=0, hist=0, psd=0, traces=0, grang=0, plotAll=0)

                # with open(filename[:-4]+'.pkl', 'w') as f:
                #     pickle.dump([sim.allSimData, sim.net, data,out], f)

            fontsiz = 14
            # raster
            if raster or plotAll:
                #timeRange = [1100, 1900] #[600, 1400]
                include = allpops+[stim] #excpops
                orderBy = ['pop', 'y']
                filename='%s%s_raster_%d_%d_%s.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
                fig1 = sim.analysis.plotRaster(include=include, timeRange=timeRange, labels='overlay', popRates=False, orderInverse=True, lw=0, markerSize=4, 
                    marker='.', popColors=popColors, showFig=0, saveFig=0, figSize=(6,8), orderBy=orderBy)
                ax = plt.gca()
                [i.set_linewidth(0.5) for i in ax.spines.itervalues()] # make border thinner
                plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')  #remove ticks
                plt.ylabel(' ') #Neurons (ordered by NCD within each pop)')
                plt.xlabel(' ')
                # plt.ylabel('Neuron number (ordered by NCD within each pop)', fontsize=fontsiz)
                # plt.xlabel('Time (ms)', fontsize=fontsiz)
                plt.subplots_adjust(left=0.17, right=0.88)
                plt.title('')
                filename='%s%s_raster_%d_%d_%s_all.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
                plt.savefig(filename, dpi=600)

            
            # spikeHist
            if spikeHist or plotAll:
                #timeRange = [1100, 1900]  #[600, 1400]
                include = [stim, ('IT2', 'IT4', 'IT5A'), 'PT5B']
                popColors[('IT2', 'IT4', 'IT5A')] = popColors['IT5A']
                fig2 = sim.analysis.plotSpikeHist(include=include, yaxis='rate', binSize=5, graphType='bar', timeRange=timeRange,  popColors=popColors, 
                     axis='off', scalebarLoc=2, showFig=False, saveFig=0, figSize=(6,3))
                ax = plt.gca()
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                plt.ylabel('Firing rate (Hz)', fontsize=fontsiz)
                plt.xlabel('Time (ms)', fontsize=fontsiz)
                plt.legend(loc = 'upper right')
                plt.subplots_adjust(left=0.12, right=0.8, bottom=0.05)
                plt.ylim(0,histMax)
                #plt.tight_layout()
                stimLabel = stim
                if stimLabel == 'TPO': stimLabel = 'PO'
                plt.legend([stimLabel,'IT2/3,4,5A', 'PT5B'])
                plt.savefig('%s%s_spikeHist_replot_%d_%d.png'%(root, simLabel, timeRange[0], timeRange[1]), dpi=600)
            

            # stats exc
            if stats or plotAll:
                #timeRange = [1100, 1900]  #[600, 1400]
                midpoint = (timeRange[1]+timeRange[0]) / 2.0
                timeRanges = [[timeRange[0], midpoint], [midpoint, timeRange[1]]]
                include = excpops
                for timeRange in timeRanges:
                    filename = root+simLabel+'_%d_%d_exc'%(timeRange[0], timeRange[1])
                    fig1 = sim.analysis.plotSpikeStats(include=include, figSize=(4,8), timeRange=timeRange, stats = ['rate'], popColors=popColors, showFig=0, saveFig=filename)
    
            # rates 
            if rates or plotAll:
                include = ['PT5B']  #['IT2', 'IT5A', 'PT5B']
                #timeRange = [1100, 1900]  #[600, 1400]
                midpoint = (timeRange[1]+timeRange[0]) / 2.0
                timeRanges = [[timeRange[0], midpoint], [midpoint, timeRange[1]]]
                colors = [popColors[inc] if inc in popColors else (0,0,0) for inc in include]
                out = sim.analysis.plotRates(include=include, timeRanges=timeRanges, ylim=[0,20], figSize=(4,2), timeRangeLabels=['Pre', 'Post'], colors=colors, showFig=0, saveFig=1)

    # seed rate stats
    if seedRates or plotAll:
        import copy

        if calculateRatesData:
            from os import listdir
            from os.path import isfile, join

            #timeRange = [1100, 1900]  #[600,1400]  ## USE [700,1300] ??
            
            jsonFolder = batchLabel 
            path = dataFolder+batchLabel #+'/noDepol'
            onlyFiles = [f for f in listdir(path) if isfile(join(path, f)) and not f.endswith('batch.json') and not f.endswith('cfg.json')]

            #outfiles = [f for f in onlyFiles if f.startswith(batchLabel+'_0') and f.endswith('.json')] 
            outfiles = [f for f in onlyFiles if f.endswith('.json')] 

            pops = ['IT5A', 'PT5B', 'IT2']
            avgs = {}
            peaks = {}            
            include = {}
            include['rates'] = pops

            with open('../sim/cells/popColors.pkl', 'r') as fileObj: popColors = pickle.load(fileObj)['popColors']
            saveData = {}
            counter=0
            rateDataFilename = dataFolder+jsonFolder+'/'+'ratesData.json'
            
            # load any existing data from ratesData.json
            try:
                with open(rateDataFilename, 'r') as fileObj: saveData = json.load(fileObj)
            except:
                pass

            # calculate and store peakRate and avgRate
            for outfile in outfiles:    
                try: 
                    filename = dataFolder+jsonFolder+'/'+outfile
                    print(filename     )

                    if outfile not in saveData:                        
                        sim,data,out=utils.plotsFromFile(filename, raster=0, stats=0, rates=1, syncs=0, hist=0, psd=0, traces=0, grang=0, plotAll=0, 
                            timeRange=timeRange, include=include, textTop='', popColors=popColors, orderBy='gid')
                        
                        for ipop, pop in enumerate(pops):
                            avgs[pop] = [float(out[2][0][ipop]), float(out[2][1][ipop])]
                            peaks[pop] = [float(out[3][0][ipop]), float(out[3][1][ipop])]

                        saveData[outfile] = {'avgs': dict(avgs), 'peaks': dict(peaks)}
                        with open(rateDataFilename, 'w') as fileObj: json.dump(saveData, fileObj)
                except:
                    pass

        # load data file
        loadData={}
        filename = dataFolder+batchLabel+'/'+'ratesData.json'
        with open(filename, 'r') as fileObj: loadData[batchLabel] = json.load(fileObj)

        # iterate data and calculate normalized % increase: (post-pre)/pre
        # calculate absolute increase in Hz (remove baseline rate) -- ih affects overall PT activity
        rows = []
        cols = ['longPop', 'pop', 'connSeed', 'stimSeed', 'ih',  'incAvg', 'incPeak']
        longPops = [ 'TPO', 'TVL', 'S2', 'M2']
        pops = ['PT5B'] 
        ihs = ['low', 'high'] #[0.3, 1.0] #[0.3, 0.4, 0.5, 1.0]
        ihOffset = 0
        rates = ['10hz', '15hz']
        irate = 0 # 0=10,1=15

        avgs = {k1: None for k1 in  pops}
        peaks = {k1: None for k1 in  pops}
        baseAvgs = None
        basePeaks = None
        fileRoot = dataFolder+batchLabel+'/'+batchLabel
        missing = []
        for iih, ih in enumerate(ihs):
            for ilong,longPop in enumerate(longPops):
                for connSeed in range(5):
                    for stimSeed in range(5):
                        try:
                            fileRoot = dataFolder+batchLabel+'/'+batchLabel
                            label = '%s_%d_%d_%d_%d.json'%(batchLabel,iih+ihOffset,connSeed,stimSeed,ilong)
                            #label = '%s_%d_%d_%d_%d_%d.json'%(batchLabel,iih+ihOffset,connSeed,stimSeed,ilong,irate)
                            
                            avgs = loadData[batchLabel][label]['avgs']
                            peaks = loadData[batchLabel][label]['peaks']

                            if longPop == 'None':
                                baseAvgs, basePeaks = copy.deepcopy(avgs), copy.deepcopy(peaks)
                            for pop in pops:
                                #avg, baseAvg = avgs[pop], baseAvgs[pop]
                                avg = avgs[pop]
                                incAvg = avg[1]  # / baseAvg[1] if baseAvg[1] > 0 else None # avg[1] - avg[0] if avg[0] > 0 else None
                                #peak, basePeak = peaks[pop], basePeaks[pop]
                                peak = peaks[pop]
                                incPeak = peak[1]  # / basePeak[1] if basePeak[1] > 0 else None  # peak[1] - peak[0] if peak[0] > 0 else None
                                rows.append([longPop, pop, connSeed, stimSeed, ih, incAvg, incPeak])
                        except Exception as e:
                            print(e)
                            missing.append(label)
        print(missing) 
        print(len(missing))
            
        # boxplots
        fontsiz = 40
        titles = {'incAvg': 'Post-stim avg rate (Hz)',
                'incPeak': 'Post-stim peak rate (Hz)'}
        df = pd.DataFrame(rows, columns=cols) 
        ihsubset = ['high', 'low']
        maxVals = {'TPO': {'incAvg': 10, 'incPeak': 47}, 'M2': {'incAvg': 9,'incPeak': 38}}
        longPop = stim
        for measure in ['incAvg', 'incPeak']:
            dflong = df.query('longPop == "%s" and ih in @ihsubset'%(longPop))
            fig=plt.figure(figsize=(8,10))
            sb.stripplot(x='ih', order=ihsubset, y=measure, data=dflong, palette='Set3', jitter=True, split=True,linewidth=1,edgecolor='gray')
            ax=sb.boxplot(x='ih', order=ihsubset, y=measure, data=dflong, palette='Set3', showfliers=False)
            #  hue='ih'

            handles, labels = ax.get_legend_handles_labels()
            #l = plt.legend(handles[0:2], labels[0:2], title='PT ih', loc=2, bbox_to_anchor=(0.96, 1), borderaxespad=0., fontsize=fontsiz)  #remove duplicate legend; 
            #plt.ylim(0,10)
            plt.tight_layout()
            plt.ylabel(titles[measure], fontsize=fontsiz)
            plt.xlabel(r'$I_h$ level', fontsize=fontsiz)
            plt.ylim(0, maxVals[stim][measure]*1.25)
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontsiz) 
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontsiz) 

            # stat sig stars
            data_low = list(df.query('ih == "low"')[measure])
            data_high = list(df.query('ih == "high"')[measure])
            z, p = scipy.stats.mannwhitneyu(data_high, data_low)
            p_value = p * 2
            s = utils.stars(p)

            try:
                y_max = maxVals[longPop][measure] # 
            except:
                y_max = max(np.percentile(data_high, 75), np.percentile(data_low, 75))
            y_min = 0
            ax.annotate("", xy=(0, y_max), xycoords='data',
                        xytext=(1, y_max), textcoords='data',
                        arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                                        connectionstyle="bar,fraction=0.2"))
            ax.text(0.5, y_max + 0.12*y_max, utils.stars(p_value),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=fontsiz) 


            fig.subplots_adjust(left=0.18, bottom=0.13)

            plt.savefig(fileRoot+'_%s_%s_%s_boxplot_onlyPT_abs.png'%(longPop, rates[irate], measure))

            


# ----------------------------------------------------------------            
def fig_simult():
    ''' Figure simultaneous input: 
        - raster plot of 0.6-1.4 sec 
        - IT5A and PT5B
    '''

    # options
    dataFolder = '../data/'
    batchLabel = 'v49_batch27'
    loadAll = 0
    simLabel = 'v49_batch27_4_4_5_0'

    allpops = ['IT2','PV2','SOM2','IT4','IT5A','PV5A','SOM5A','IT5B','PT5B','PV5B','SOM5B','IT6','CT6','PV6','SOM6']
    excpops = ['IT2','IT4','IT5A','IT5B','PT5B','IT6','CT6']
    inhpops = ['SOM2','PV2', 'SOM5A','PV5A',  'SOM5B','PV5B',  'SOM6', 'PV6']
    excpopsu = ['IT2','IT4','IT5A','PT5B']

    with open('../sim/cells/popColors.pkl', 'r') as fileObj: popColors = pickle.load(fileObj)['popColors']
    popColors['S2'] = [0.90,0.76,0.00]
    popColors['M2'] = [0.42,0.67,0.9]

    plt.style.use('seaborn-ticks') 

    # load data
    root = dataFolder+batchLabel+'/'
    
    raster = 1
    simult = 0
    plotAll = 0

    filename = root+simLabel+'.json'
    sim,data,out = utils.plotsFromFile(filename, raster=0, stats=0, rates=0, syncs=0, hist=0, psd=0, traces=0, grang=0, plotAll=0)

        # with open(filename[:-4]+'.pkl', 'w') as f:
        #     pickle.dump([sim.allSimData, sim.net, data,out], f)

    fontsiz = 14
    # raster
    if raster or plotAll:
        timeRange = [600, 1400]
        include = allpops+['S2','M2'] #excpops
        orderBy = ['pop', 'y']
        filename='%s%s_raster_%d_%d_%s.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
        fig1 = sim.analysis.plotRaster(include=include, timeRange=timeRange, labels='overlay', popRates=False, orderInverse=True, lw=0, markerSize=4, 
            marker='.', popColors=popColors, showFig=0, saveFig=0, figSize=(6,8), orderBy=orderBy)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.ylabel('Neuron number', fontsize=fontsiz)
        plt.xlabel('Time (ms)', fontsize=fontsiz)
        plt.subplots_adjust(left=0.17, right=0.88)
        plt.title('')
        filename='%s%s_raster_%d_%d_%s_all.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
        plt.savefig(filename, dpi=600)


    if simult or plotAll:
        import copy
        # load
        filename = dataFolder+batchLabel+'/'+'ratesData.json'
        with open(filename, 'r') as fileObj: loadData = json.load(fileObj)

        # iterate data and calculate normalized % increase: (post-pre)/pre
        # calculate absolute increase in Hz (remove baseline rate) -- ih affects overall PT activity
        rows = []
        cols = ['long2Pop', 'pop', 'interval', 'ih',  'incAvg', 'incPeak']
        long2Pops = ['TPO+TVL', 'TVL+TPO', 'TVL+S2', 'S2+TVL', 'S2+M2', 'M2+S2'] 
        pops = ['IT2', 'IT5A', 'PT5B'] 
        intervals = list(np.arange(1000, 1220, 20))
        ihs = ['high', 'low']

        avgs = {k1:{k2:None for k2 in ihs} for k1 in pops}
        peaks = {k1:{k2:None for k2 in ihs} for k1 in pops}

        fileRoot = dataFolder+batchLabel+'/'+batchLabel
        missing = []
        for ilong,long2Pop in enumerate(long2Pops):
            for iinterval,interval in enumerate(intervals):
                # try:
                label = '%s_%d_%d_%d_1.json'%(fileRoot,ilong,ilong,iinterval)
                print(label)
                [avgs['PT5B']['low'], peaks['PT5B']['low'],\
                 avgs['PT5B']['high'], peaks['PT5B']['high'], \
                 avgs['IT5A']['low'], peaks['IT5A']['low'], \
                 avgs['IT5A']['high'], peaks['IT5A']['high'],\
                 avgs['IT2']['low'], peaks['IT2']['low'], \
                 avgs['IT2']['high'], peaks['IT2']['high']] = loadData[label]

                for pop in pops:
                    for ih in ihs:
                        avg = avgs[pop][ih]
                        incAvg = (avg[1]) # / baseAvg[1] if baseAvg[1] > 0 else None # avg[1] - avg[0] if avg[0] > 0 else None
                        peak = peaks[pop][ih]
                        incPeak = (peak[1]) # / basePeak[1] if basePeak[1] > 0 else None  # peak[1] - peak[0] if peak[0] > 0 else None
                        rows.append([long2Pop, pop, interval, ih, incAvg, incPeak])
                # except Exception as e:
                #     print(e)
                #    missing.append(label)
        print(missing)
        print(len(missing))
        
        # boxplots
        fontsiz = 19
        long2PopGroups = [['S2+M2', 'M2+S2']]  #['TPO+TVL', 'TVL+TPO'], ['TVL+S2', 'S2+TVL'],  
        labels = ['M2+S2, high Ih',
                'S2+M2, high Ih',
                'M2+S2, low Ih',
                'S2+M2, low Ih']
        titles = {'incAvg': 'Post-stim avg rate (Hz)',
                'incPeak': 'Post-stim peak rate (Hz)'}
        df = pd.DataFrame(rows, columns=cols) 
        for pop in pops:
            for long2PopGroup in long2PopGroups:
                for measure in ['incAvg', 'incPeak']:
                    dflong = df.query('pop==@pop and long2Pop == @long2PopGroup')
                    dflong['interval']=[x-1000 for x in dflong['interval']]
                    dfpiv = pd.pivot_table(dflong, index='interval', columns=['ih', 'long2Pop'], values=measure)
                    M2 = popColors['PT5B']
                    S2 = popColors['IT5A']
                    dfpiv.plot(color= [M2,S2,M2,S2], style=['--','--','-','-'], marker='o', xticks=range(0,220,20))
                    plt.legend('off')
                    #plt.legend(labels, title="", fontsize=fontsiz-4, loc='upper left', bbox_to_anchor=(0.3, 1.2),  
                    #    shadow=True, ncol=2)
                    
                    plt.tight_layout()
                    plt.ylabel(titles[measure], fontsize=fontsiz)
                    plt.xlabel('Interval (ms)', fontsize=fontsiz)
                    #plt.ylim(0,38)
                    ax = plt.gca()
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(fontsiz) 
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(fontsiz) 
                    fig=plt.gcf()
                    fig.subplots_adjust(left=0.15, bottom=0.15,right=0.9, top=0.99)
                    plt.savefig(fileRoot+'_%s_%s_%s_plot.png'%(pop, long2PopGroup, measure), dpi=300)


# ----------------------------------------------------------------
def fig_long():
    ''' Figure spontaneous activity: 
        - raster plot of 3-4 sec 
        - spike stats 1-5 sec, exc + inh, rate+isicv
        - rate PSD 1-5 sec, all, IT2, IT5A, PT5B
        - Granger 1-5 sec, IT2<->IT5A, IT2<->PT5B, IT5A<->PT5B '''

    # options
    dataFolder = '../data/'
    batchLabel = 'v50_batch1'
    loadAll = 0
    simLabels = ['v50_batch1_%d_%d_%d_%d_%d_%d_%d_%d'%(x,x,x,x,x,x,x,y) for x in [2] for y in [0,3]] 

    allpops = ['IT2','PV2','SOM2','IT4','IT5A','PV5A','SOM5A','IT5B','PT5B','PV5B','SOM5B','IT6','CT6','PV6','SOM6']
    excpops = ['IT2','IT4','IT5A','IT5B','PT5B','IT6','CT6']
    inhpops = ['SOM2','PV2', 'SOM5A','PV5A',  'SOM5B','PV5B',  'SOM6', 'PV6']
    excpopsu = ['IT2','IT4','IT5A','PT5B']

    with open('../sim/cells/popColors.pkl', 'r') as fileObj: popColors = pickle.load(fileObj)['popColors']
    popColors['S2'] = [0.90,0.76,0.00]
    popColors['M2'] = [0.42,0.67,0.9]

    plt.style.use('seaborn-ticks') 

    # load data
   
    raster = 1
    psd = 0
    grang = 0
    plotAll = 0

    for simLabel in simLabels:
        root = dataFolder+batchLabel+'/'
        filename = root+simLabel+'.json'
        sim,data,out = utils.plotsFromFile(filename, raster=0, stats=0, rates=0, syncs=0, hist=0, psd=0, traces=0, grang=0, plotAll=0)

        # raster
        if raster or plotAll:
            timeRange = [1000, 5000]
            include = ['S2'] # + excpops #, 'IT2', 'IT5A', 'PT5B']
            orderBy = 'pop'
            #filename = '%s%s_raster_%d_%d_%s.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
            fig1 = sim.analysis.plotRaster(include=include, timeRange=timeRange, labels='overlay', popRates=False, orderInverse=True, lw=0, markerSize=4, 
                marker='.', popColors=popColors, showFig=0, saveFig=0, figSize=(7,5)) #, orderBy=orderBy)# ax = plt.gca()
            ax = plt.gca()
            plt.ylabel('Neuron number')
            plt.title('')
            filename='%s%s_raster_%d_%d_%s.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
            plt.savefig(filename, dpi=400)


        # psd
        fontsiz=24
        figsiz = (15,6)
        if psd or plotAll:
            timeRange = [1000, 5000]
            include = ['allCells', 'IT2', 'IT5A', 'PT5B', 'S2', 'M2']
            popColors['allCells'] = 'k'
            fig3 = sim.analysis.plotRatePSD(include=include, timeRange=timeRange, Fs=160, smooth=16 , showFig=0, saveFig=0, popColors=popColors, figSize=figsiz)
            # ylim=[-55, 0]
            ax = plt.gca()
            plt.xlabel(' ', fontsize=fontsiz) #Frequency (Hz)', fontsize=fontsiz)
            plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=fontsiz)
            ax = plt.gca()
            plt.setp(ax.get_xticklabels(),fontsize=fontsiz-2)
            plt.setp(ax.get_yticklabels(),fontsize=fontsiz-2)
            plt.subplots_adjust(bottom=0.1, left=0.1, right=0.98)
            #plt.tight_layout()
            plt.legend(fontsize=fontsiz)
            plt.savefig('%s%s_spikePSD_%d_%d.png'%(root, simLabel, timeRange[0], timeRange[1]), dpi=300)

        # granger
        if grang or plotAll:
            figsiz = (8,6)
            fontsiz = 20
            timeRange = [1000, 5000]
            print('Calculating and plotting Granger...')
            sim.allSimData = data['simData']
            spkts = sim.allSimData['spkt']
            spkids = sim.allSimData['spkid']
            spkids,spkts = zip(*[(spkid,spkt-timeRange[0]) for spkid,spkt in zip(spkids,spkts) if timeRange[0] <= spkt <= timeRange[1]])

            popLabels = sim.net.allPops.keys()
            gidPops = [cell['tags']['pop'] for cell in sim.net.allCells]
            popNumCells = [float(gidPops.count(pop)) for pop in popLabels]
            spktPop = {}
            for pop, popNum in zip(popLabels, popNumCells):
                spktPop[pop] = [spkt for spkt,spkid in zip(spkts,spkids) if sim.net.allCells[int(spkid)]['tags']['pop']==pop]

            binsiz = 5
            F, Fx2y, Fy2x, Fxy = utils.granger(spktPop['IT2'], spktPop['PT5B'], binSize=binsiz)
            F_b, Fx2y_b, Fy2x_b, Fxy_b = utils.granger(spktPop['IT2'], spktPop['IT5A'], binSize=binsiz)
            F_b2, Fx2y_b2, Fy2x_b2, Fxy_b2 = utils.granger(spktPop['IT5A'], spktPop['PT5B'], binSize=binsiz)
            
            figh = plt.figure(figsize=figsiz)
            plt.plot(F_b, Fy2x_b, 'r-', label = 'IT2 -> IT5A')
            plt.plot(F_b, Fx2y_b, 'r:', label = 'IT5A -> IT2')
            plt.plot(F, Fy2x, 'b-', label = 'IT2 -> PT5B')
            plt.plot(F, Fx2y, 'b:', label = 'PT5B -> IT2')
            plt.plot(F, Fy2x_b2, 'm-', label = 'IT5A -> PT5B')
            plt.plot(F, Fx2y_b2, 'm:', label = 'PT5B -> IT5A')


            plt.legend(fontsize=fontsiz)
            plt.ylim(0,1.5)
            plt.xlim(0,80)
            ax = plt.gca()
            plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
            plt.ylabel('Granger Causality', fontsize=fontsiz)
            plt.setp(ax.get_xticklabels(),fontsize=fontsiz-2)
            plt.setp(ax.get_yticklabels(),fontsize=fontsiz-2)
            plt.subplots_adjust(bottom=0.2, left=0.15, right=0.98)
            #plt.tight_layout()
            #plt.savefig(filename+'_Granger_%d_%d.png'%(timeRange[0], timeRange[1]), dpi=300)
            plt.savefig('%s%s_Granger_%d_%d_local.png'%(root, simLabel,timeRange[0], timeRange[1]), dpi=300)


            # S2 <-> IT2, IT5A, PT5B
            binsiz = 5
            F, Fx2y, Fy2x, Fxy = utils.granger(spktPop['S2'], spktPop['IT2'], binSize=binsiz)
            F_b, Fx2y_b, Fy2x_b, Fxy_b = utils.granger(spktPop['S2'], spktPop['IT5A'], binSize=binsiz)
            F_b2, Fx2y_b2, Fy2x_b2, Fxy_b2 = utils.granger(spktPop['S2'], spktPop['PT5B'], binSize=binsiz)
            
            figh = plt.figure(figsize=figsiz)
            plt.plot(F, Fy2x, 'r-', label = 'S2 -> IT2')
            plt.plot(F, Fx2y, 'r:', label = 'IT2 -> S2')
            plt.plot(F_b, Fy2x_b, 'm-', label = 'S2 -> IT5A')
            plt.plot(F_b, Fx2y_b, 'm:', label = 'IT5A -> S2')
            plt.plot(F, Fy2x_b2, 'b-', label = 'S2 -> PT5B')
            plt.plot(F, Fx2y_b2, 'b:', label = 'PT5B -> S2')

            plt.legend(fontsize=fontsiz-4)
            plt.ylim(0,1.0)
            plt.xlim(0,80)
            ax = plt.gca()
            plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
            plt.ylabel('Granger Causality', fontsize=fontsiz)
            plt.setp(ax.get_xticklabels(),fontsize=fontsiz-2)
            plt.setp(ax.get_yticklabels(),fontsize=fontsiz-2)
            plt.subplots_adjust(bottom=0.2, left=0.15, right=0.98)
            #plt.tight_layout()
            plt.savefig('%s%s_Granger_%d_%d_long_S2.png'%(root, simLabel,timeRange[0], timeRange[1]), dpi=300)


            # M2 <-> IT2, IT5A, PT5B
            binsiz = 5
            F, Fx2y, Fy2x, Fxy = utils.granger(spktPop['M2'], spktPop['IT2'], binSize=binsiz)
            F_b, Fx2y_b, Fy2x_b, Fxy_b = utils.granger(spktPop['M2'], spktPop['IT5A'], binSize=binsiz)
            F_b2, Fx2y_b2, Fy2x_b2, Fxy_b2 = utils.granger(spktPop['M2'], spktPop['PT5B'], binSize=binsiz)
            
            figh = plt.figure(figsize=figsiz)
            plt.plot(F, Fy2x, 'r-', label = 'M2 -> IT2')
            plt.plot(F, Fx2y, 'r:', label = 'IT2 -> M2')
            plt.plot(F_b, Fy2x_b, 'm-', label = 'M2 -> IT5A')
            plt.plot(F_b, Fx2y_b, 'm:', label = 'IT5A -> M2')
            plt.plot(F, Fy2x_b2, 'b-', label = 'M2 -> PT5B')
            plt.plot(F, Fx2y_b2, 'b:', label = 'PT5B -> M2')

            plt.legend(fontsize=fontsiz-4)
            plt.ylim(0,0.4)
            plt.xlim(0,80)
            ax = plt.gca()
            plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
            plt.ylabel('Granger Causality', fontsize=fontsiz)
            plt.setp(ax.get_xticklabels(),fontsize=fontsiz-2)
            plt.setp(ax.get_yticklabels(),fontsize=fontsiz-2)
            plt.subplots_adjust(bottom=0.2, left=0.15, right=0.98)
            #plt.tight_layout()
            plt.savefig('%s%s_Granger_%d_%d_long_M2.png'%(root, simLabel,timeRange[0], timeRange[1]), dpi=300)


def fig_conn():
    # NOTE: data files need to be loaded using Python 2!
    # load conn matrices
    # with open('../data/v53_manualTune/v53_tune7_conn_strength_conn.pkl', 'rb') as f:
    #     data = pickle.load(f)

    simLabel = batchLabel = 'v53_manualTune'
    dataFolder = '../data/'
    root = dataFolder+batchLabel+'/'

    with open('../data/v53_manualTune/v53_tune7_conn_weight_all_py3.pkl', 'rb') as f:
        dataW = pickle.load(f)

    with open('../data/v53_manualTune/v53_tune7_conn_probability_all_py3.pkl', 'rb') as f:
        dataP = pickle.load(f)

    
    popsPre = dataP['includePre']
    popsPost = dataP['includePost']
    
    # strength
    # connMatrix = dataW['connMatrix'].T * dataP['connMatrix'].T
    # feature = 'strength'

    # prob
    connMatrix = dataP['connMatrix'].T
    feature = 'Probability of connection'

    # weight
    # connMatrix = dataW['connMatrix'].T
    # feature = 'weight'

    excPopsInds = [0, 3, 4, 7, 8, 11, 12]
    inhPopsInds = [1, 2, 5, 6, 9, 10, 13, 14]


    connMatrixE = connMatrix[:,excPopsInds]
    connMatrixI = connMatrix[:,inhPopsInds]

    popsPreE = [popsPre[i] for i in excPopsInds]
    popsPreI = [popsPre[i] for i in inhPopsInds]

    # font
    fontsiz = 18
    plt.rcParams.update({'font.size': fontsiz})


    # # ----------------------- 
    # # conn matrix full
    # # connMatrix[:, inhPopsInds] *= -1.0

    # vmin = np.nanmin(connMatrix)
    # vmax = np.nanmax(connMatrix)
    
    # # viridisBig = cm.get_cmap('viridis', 512)
    # # blue = viridisBig(np.linspace(0, 0.5, 256))
    # # yellow = viridisBig(np.linspace(0.5, 1.0, int(256.0 * vmax/abs(vmin))))
    

    # # newcolors = np.vstack((blue, yellow))
    # # newcmap = ListedColormap(newcolors)
    
    # # cmap1 = bicolormap(gap=0, mingreen=0, redbluemix=1, epsilon=0)
    # # cmap2 = bicolormap(gap=0, mingreen=0, redbluemix=0, epsilon=0.1)
    # # cmap3 = bicolormap(gap=0.3, mingreen=0.2, redbluemix=0, epsilon=0.01)
    # # cmap4 = bicolormap(gap=0.0, mingreen=0.5, redbluemix=0.5, epsilon=0.0)

    # plt.figure(figsize=(12, 12))
    # h = plt.axes()
    # plt.imshow(connMatrix, interpolation='nearest', cmap='viridis', vmin=vmin, vmax=vmax)  #_bicolormap(gap=0)

    # for ipop, pop in enumerate(popsPost):
    #     plt.plot(np.array([0,len(popsPre)])-0.5,np.array([ipop,ipop])-0.5,'-',c=(0.7,0.7,0.7))
    # for ipop, pop in enumerate(popsPre):
    #     plt.plot(np.array([ipop,ipop])-0.5,np.array([0,len(popsPost)])-0.5,'-',c=(0.7,0.7,0.7))

    # # Make pretty
    # h.set_xticks(list(range(len(popsPre))))
    # h.set_yticks(list(range(len(popsPost))))
    # h.set_xticklabels(popsPre, rotation=45)
    # h.set_yticklabels(popsPost)
    # h.xaxis.set_ticks_position('top')
    # plt.xlim(-0.5,len(popsPre)-0.5)
    # plt.ylim(len(popsPost) - 0.5, -0.5)

    # plt.grid(False)
    
    # clim = [np.nanmin(connMatrix), np.nanmax(connMatrix)]
    # plt.clim(clim[0], clim[1])
    # plt.colorbar(label=feature, shrink=0.8) #.set_label(label='Fitness',size=20,weight='bold')
    # plt.ylabel('Target population')
    # h.xaxis.set_label_coords(0.5, 1.10)
    # plt.xlabel('Source population')
    # plt.title('Connection ' + feature + ' matrix', y=1.11, fontWeight='bold')

    # plt.savefig('%s%s_connFull_orig_%s.png' % (root, simLabel, feature), dpi=300)
    # plt.show()


    # ----------------------- 
    # conn matrix E
    plt.figure(figsize=(7.5, 12))
    h = plt.axes()
    plt.imshow(connMatrixE, interpolation='nearest', cmap='viridis', vmin=np.nanmin(connMatrixE), vmax=np.nanmax(connMatrixE))  #_bicolormap(gap=0)

    for ipop, pop in enumerate(popsPost):
        plt.plot(np.array([0,len(popsPreE)])-0.5,np.array([ipop,ipop])-0.5,'-',c=(0.7,0.7,0.7))
    for ipop, pop in enumerate(popsPreE):
        plt.plot(np.array([ipop,ipop])-0.5,np.array([0,len(popsPost)])-0.5,'-',c=(0.7,0.7,0.7))

    # Make pretty
    h.set_xticks(list(range(len(popsPreE))))
    h.set_yticks(list(range(len(popsPost))))
    h.set_xticklabels(popsPreE, rotation=45)
    h.set_yticklabels(popsPost)
    h.xaxis.set_ticks_position('top')
    plt.xlim(-0.5,len(popsPreE)-0.5)
    plt.ylim(len(popsPost) - 0.5, -0.5)

    plt.grid(False)
    
    clim = [np.nanmin(connMatrixE), np.nanmax(connMatrixE)]
    plt.clim(clim[0], clim[1])
    plt.colorbar(label=feature, shrink=0.8) #.set_label(label='Fitness',size=20,weight='bold')
    plt.ylabel('Target population')
    h.xaxis.set_label_coords(0.5, 1.12)
    plt.xlabel('Source population')
    #plt.title('Connection ' + feature + ' matrix', y=1.12, fontWeight='bold')
    plt.subplots_adjust(left = 0.2, top =(0.9), bottom=(0.05))

    plt.savefig('%s%s_connE_%s.png'%(root, simLabel, feature), dpi=300)
    
    # ---------------
    # conn matrix I
    plt.figure(figsize=(9.0,12))
    h = plt.axes()
    plt.imshow(connMatrixI, interpolation='nearest', cmap='viridis', vmin=np.nanmin(connMatrixI), vmax=np.nanmax(connMatrixI))  #_bicolormap(gap=0)

    for ipop, pop in enumerate(popsPost):
        plt.plot(np.array([0,len(popsPreI)])-0.5,np.array([ipop,ipop])-0.5,'-',c=(0.7,0.7,0.7))
    for ipop, pop in enumerate(popsPreI):
        plt.plot(np.array([ipop,ipop])-0.5,np.array([0,len(popsPost)])-0.5,'-',c=(0.7,0.7,0.7))

    # Make pretty
    h.set_xticks(list(range(len(popsPreI))))
    h.set_yticks(list(range(len(popsPost))))
    h.set_xticklabels(popsPreI, rotation=45)
    h.set_yticklabels(popsPost)
    h.xaxis.set_ticks_position('top')
    plt.xlim(-0.5,len(popsPreI)-0.5)
    plt.ylim(len(popsPost) - 0.5, -0.5)

    plt.grid(False)
    
    clim = [np.nanmin(connMatrixI), np.nanmax(connMatrixI)]
    plt.clim(clim[0], clim[1])
    plt.colorbar(label=feature, shrink=0.8) #.set_label(label='Fitness',size=20,weight='bold')
    plt.xlabel('Target population')
    h.xaxis.set_label_coords(0.5, 1.12)
    plt.ylabel('Source population')
    #plt.title('Connection ' + feature + ' matrix', y=1.12, fontWeight='bold')
    plt.subplots_adjust(left=0.2, top=(0.9), bottom=(0.05))

    plt.savefig('%s%s_connI_%s.png'%(root, simLabel, feature), dpi=300)

    #plt.show()




# Main code
if __name__ == '__main__': 
    
    # fig_fIcurves()
    sim = fig_spont()
    # sim = fig_osc_lfp()
    # sim = fig_osc_spikes()
    # sim = fig_lfp_ih()
    # sim = fig_psd()
    # """  """sim = fig_granger()
    
    # stim at 1000
    # df = fig_pulse(batchLabel = 'v53_batch14', simLabels = ['v53_batch14_0_0_0_0', 'v53_batch14_1_0_0_0'], stim = 'TPO', timeRange= [900,1200], histMax = 32)  # fig 7
    #df = fig_pulse(batchLabel = 'v53_batch14', simLabels = ['v53_batch14_0_0_0_3', 'v53_batch14_1_0_0_3'], stim = 'M2', timeRange= [900,1200], histMax = 32)  # fig 7

    # stim at 1500
    # df = fig_pulse(batchLabel = 'v53_batch14', simLabels = ['v53_batch15_0_0_0_2_1', 'v53_batch15_1_0_0_2_1'], stim = 'S2', histMax = 50)  # fig 6 

    # df=fig_simult()
    # fig_long()

    # fig_conn()
    