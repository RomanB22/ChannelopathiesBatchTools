#/* $Id: modindex.py,v 1.2 2012/10/03 20:03:16 mohdsh Exp $ */
# Written by Mohamed Sherif (Mohamed.Sherif@yale.edu)
'''This module contains functions that would enable the calculation of modulation
index or measure of cross-frequency coupling (CFC) between the phase of slow frequency
and the amplitude (or envelope) of faster frequency.
The algorithm is based on the following article:
Measuring Phase-Amplitude Coupling Between Neuronal Oscillations
of Different Frequencies
Adriano B. L. Tort, Robert Komorowski, Howard Eichenbaum, and Nancy Kopell
J Neurophysiol 104: 1195-1210, 2010, and the Matlab code that A. Tort supplied us.

A set of functions uses variable bandwidth for filtering the fast frequency (frequency for amplitude). This is supported by:
Variable Bandwidth Filtering for Improved Sensitivity of Cross-Frequency Coupling Metrics.
Berman J, McDaniel J, Liu S, Cornew L, Gaetz W, Roberts TP, Edgar JC. 
Brain Connect. 2012 May 13. [Epub ahead of print] PubMed PMID: 22577870.
#####################################################################################
The following is a suggested step-wise approach when assessing CFC.
1 - Identify the peaks of CFC.
2 - Identify the preferred phase(s) of the slow frequency on which the fast frequency rides.
3 - Since the CFC is non-stationary (according to our results so far), assess how the strength of CFC changes over time.
#####################################################################################
The following would be an example for the commands to run in order:
# in a terminal
ipython -pylab
from modIndexList import *
#mylfp is a hoc vector (after subtraction of mean) of 5 seconds duration, with sampling rate of 3030
# calculate modulation index array for slow frequency between 5 & 10 Hz, and for fast frequency between 20 & 100 Hz.
# going in steps of 1 Hz for the slow frequency, in steps of 5 Hz for the fast frequency. 2 Hz is the bandwidth used 
# to filter the slow frequency. No need to decide what is the bandwidth for the fast frequency using this function. It will
# be defined as double the slow frequency. the 360 degrees cycle will be divided into 18 bins.
phaseFreq, ampFreq, MIarr = varModIndArr(mylfp, 3030, 1, 4, 5, 10, 20, 100, 1, 5, 2, 18)


sample use:
modIndex(raw_signal, samplingRate, f_phaseMin, f_phaseMax, f_ampMin, f_ampMax, phaseBins) #default phaseBins is 100

raw_signal is 1-dnarray

it is based on the matlab code by Tort

'''

from neuron import *
from ctypes import *
import numpy as np
h.load_file('nqs.hoc')

# load the C dll for use in calc modulation index
# written by sam neymotin - uses modind.c code for faster calculation
def loadmodfunc ():
  from os import system
  modfunc = None
  try:
    modfunc = cdll.LoadLibrary("./modind.so")
    modfunc.modindex.restype = c_double
    modfunc.meanampbins.restype = c_double
  except:
    print('compiling modind.c')
    system('gcc -Wall -fPIC -c modind.c')
    system('gcc -shared -o modind.so modind.o')
    modfunc = cdll.LoadLibrary("./modind.so")
    modfunc.modindex.restype = c_double
    modfunc.meanampbins.restype = c_double
  return modfunc

modfunc = loadmodfunc()

import numpy
from math import log, pi, ceil
from filt import gethilbnq #/usr/site/nrniv/local/python/filt.py
#hilbert function has a phase range of -pi to pi
from pylab import bar, contourf, xlabel, ylabel, show, axis, xticks, colorbar, title, plot, contour
from time import time, clock
import random
import multiprocessing

from filter import bandpass
from scipy.signal import lfilter, hilbert, butter

#
def phaseFreqTransformed (raw_signal, samplingRate, from_t, to_t, f_phaseMin, f_phaseMax, step, bandWidth):
    '''raw_signal is hoc vector. from_t & to_t are both in seconds. The function will work on the segment
    from_t TO to_t plus 1 second on each side. It will filter between values from f_phaseMin to f_phaseMax in steps of step,
    with bandWidth. 1 second is then removed from the filtered signal, and the segment from_t TO to_t is returned
    as a transormed array: each row is the filtered signal in a particular step range of freq; the columns are the
    values for each point in time'''
    #proceed if raw_signal is longer than 2 seconds, there is at least 1 second before from_t, and there is at least 1 second after
    if checkSize(raw_signal,samplingRate,from_t,to_t):
        #time0, clock0 = time(), clock()
        phaseFreqVec = h.Vector()
        phaseFreqVec.indgen(f_phaseMin, f_phaseMax, step)
        mysignal = raw_signal.to_python()[int((from_t-1)*samplingRate):int((to_t+1)*samplingRate-1)]
        mylist = [] # to collect all time series
        #print('Calculating transformed phase array...')
        for elem in phaseFreqVec:#will go through each step, calculating within bandWidth
            Pf1 = max(0.0,elem - bandWidth / 2.0)
            Pf2 = elem + bandWidth / 2.0
            sig = bandpass(mysignal, Pf1, Pf2, df = samplingRate, zerophase = True)
            H = hilbert(sig)
            phaseFreq = h.Vector()
            phaseFreq.from_python(numpy.angle(H)) 
            #remove last 1 second
            last_1s = [(phaseFreq.size()-1 - samplingRate*1), phaseFreq.size()-1]
            phaseFreq.remove(last_1s[0], last_1s[1])
            #remove first 1 second
            first_1s = [0, samplingRate*1]
            phaseFreq.remove(first_1s[0], first_1s[1])     
            mylist.append(phaseFreq)            
        #print('phaseFreqTransformed time:', time()-time0, 'clock:', clock()-clock0)
        #print('transformed phase array size:', len(mylist))
        return mylist
    else: 
        return None

#      
def ampFreqTransformed(raw_signal, samplingRate, from_t, to_t, f_ampMin, f_ampMax, step, bandWidth):
    '''raw_signal is a hoc vector. from_t & to_t are both in seconds. The function will work on the segment
    from_t TO to_t plus 1 second on each side. It will filter between values from f_ampMin to f_ampMax in steps of step,
    with bandWidth. 1 second is then removed from the filtered signal, and the segment from_t TO to_t is returned
    as a transormed array: each row is the filtered signal in a particular step range of freq; the columns are the
    values for each point in time'''
    if checkSize(raw_signal,samplingRate,from_t,to_t):
        #time0, clock0 = time(), clock()
        ampFreqVec = h.Vector()
        ampFreqVec.indgen(f_ampMin, f_ampMax, step)
        mysignal = raw_signal.to_python()[int((from_t-1)*samplingRate):int((to_t+1)*samplingRate-1)]
        mylist = []
        print('Calculating transformed amplitude array...')
        for elem in ampFreqVec:
            Af1 = max(0.0,elem - bandWidth / 2.0)
            Af2 = elem + bandWidth / 2.0
            sig = bandpass(mysignal, Af1, Af2, df = samplingRate, zerophase = True)
            H = hilbert(sig)
            ampFreq = h.Vector()
            ampFreq.from_python(abs(H))             
            #remove last 1 second
            last_1s = [(ampFreq.size()-1 - samplingRate*1), ampFreq.size()-1]
            ampFreq.remove(last_1s[0], last_1s[1])
            #remove first 1 second
            first_1s = [0, samplingRate*1]
            ampFreq.remove(first_1s[0], first_1s[1])
            mylist.append(ampFreq)
        #print('ampFreqTransformed time:', time()-time0, 'clock:', clock()-clock0)
        #print('transformed amp array size:', len(mylist))
        return mylist    
    else: 
        return None

def listBins(numBins):
    '''a 2*pi signal is divided into numBins. return a vector of the bins'''
    binVec = h.Vector(numBins)
    binSize = 2 * pi / numBins
    binVec.indgen(-pi, binSize)
    return binVec

#
def meanAmpBins (phaseVec, ampVec, listBins):
    '''input: 2 vectors, each of filtered signal through time, one for phase & other for amplitude; list containing
    the bins of the phase. Returns vector of the values of the mean amp at different bins of phase'''
    numBins = len(listBins)
    meanAmpBins = h.Vector(numBins)
    binSize = 2 * pi / numBins
    meanAmpBins.resize(0)
    indexVec = h.Vector()
    for i in range(numBins):
      #index from phaseVec where values lies between listBins[i] and listBins[i]+binSize        
      indexVec.indvwhere(phaseVec, "[)", listBins[i], listBins[i] + binSize)
      amplitude = ampVec.ind(indexVec)#vector of values from ampVec corresponding to indices from phaseVec
      if amplitude.size()>0:
        meanAmp = amplitude.mean()
        meanAmpBins.append(meanAmp)
      else:
        meanAmpBins.append(0.0)
    return meanAmpBins

#
def modIndex (meanAmpBins):
    '''input is a vector of mean amplitudes calculated for each phase bin, generated
    by meanAmpBins, and return MI for that vector'''
    num = len(meanAmpBins)
    sumMean = meanAmpBins.sum()
    nMeanAmp = h.Vector()
    nMeanAmp.copy(meanAmpBins)
    nMeanAmp.div(sumMean)
    sigma = 0 # calculate entropy
    for amp in nMeanAmp:
        if amp > 0.0: sigma -= (amp * log(amp))
    return 1.0 - sigma / log(num)

#
def modIndArr (raw_signal, sampr, from_t, to_t, f_phaseMin, f_phaseMax, f_ampMin, f_ampMax,\
              phaseStep=1, phase_bandWidth=2, ampStep=5, amp_bandWidth=10, nBins=18, NShuffle=10, Z_score=False):
  '''will calculate modulation index array and return it, with the arrays to plot on x-axis
  (freq for phase) and y-axis (freq for amp)'''
  if checkSize(raw_signal,sampr,from_t,to_t):

    import IPython; IPython.embed()


    phaseFtrf = phaseFreqTransformed(raw_signal, sampr, from_t, to_t, f_phaseMin, f_phaseMax, phaseStep, phase_bandWidth)
    ampFtrf = ampFreqTransformed(raw_signal, sampr, from_t, to_t, f_ampMin, f_ampMax, ampStep, amp_bandWidth)
    Bins_list = listBins(nBins)
    modArr = numpy.zeros((len(ampFtrf), len(phaseFtrf)))
    print('Calculating MI...')
    #time0, clock0 = time(), clock()
    meanAmpBinsVec = h.Vector(nBins)
    mabp = meanAmpBinsVec.as_numpy().ctypes.data_as(c_void_p)
    binsp = Bins_list.as_numpy().ctypes.data_as(c_void_p)
    for column in range(int(len(phaseFtrf))):
      phaseSlice = phaseFtrf[column]#.getrow(column)
      psp = phaseSlice.as_numpy().ctypes.data_as(c_void_p)
      pspsz = len(phaseSlice)
      for row in range(int(len(ampFtrf))):
        ampSlice=ampFtrf[row]#.getrow(row)
        modArr[row,column]=modfunc.meanampbins(psp,ampSlice.as_numpy().ctypes.data_as(c_void_p),pspsz,binsp,int(nBins),mabp)
    #print('modIndArr loop time:', time()-time0, 'clock:', clock()-clock0)
    phaseFreq = numpy.r_[f_phaseMin:f_phaseMax+1:phaseStep]
    ampFreq = numpy.r_[f_ampMin:f_ampMax + 1:ampStep]
    
    # calculate Z-score based on shuffled signal

    if Z_score:
        shuffledList = []
        #signalShuffled = np.array(list(raw_signal))
        phaseFtrfShuffled = [h.Vector((np.array(x))) for x in phaseFtrf]
        for ishuffle in range(NShuffle):
            print('Shuffle %d...'%(ishuffle))
            
            # for each calculate max Granger value (starting at freq index 1) 

            # method 1 - shuffling lfp signal lead to relatively high modulation indices
            # np.random.shuffle(signalShuffled)
            # phaseFreq, ampFreq, modArr = modIndArr (h.Vector(signalShuffled), sampr, from_t, to_t, f_phaseMin, f_phaseMax, f_ampMin, f_ampMax,\
            #   phaseStep, phase_bandWidth, ampStep, amp_bandWidth, nBins, NShuffle, Z_score=False)

            #method 2

            tmp = [np.array(x) for x in phaseFtrfShuffled]
            for x in tmp: np.random.shuffle(x)
            phaseFtrfShuffled = [h.Vector(x) for x in tmp]
            
            modArrShuffle = numpy.zeros((len(ampFtrf), len(phaseFtrfShuffled)))
            print('Calculating shuffled MI...')
            #time0, clock0 = time(), clock()
            meanAmpBinsVec = h.Vector(nBins)
            mabp = meanAmpBinsVec.as_numpy().ctypes.data_as(c_void_p)
            binsp = Bins_list.as_numpy().ctypes.data_as(c_void_p)
            for column in range(int(len(phaseFtrfShuffled))):
                phaseSlice = phaseFtrfShuffled[column]#.getrow(column)
                psp = phaseSlice.as_numpy().ctypes.data_as(c_void_p)
                pspsz = len(phaseSlice)
                for row in range(int(len(ampFtrf))):
                    ampSlice=ampFtrf[row]#.getrow(row)
                    modArrShuffle[row,column]=modfunc.meanampbins(psp,ampSlice.as_numpy().ctypes.data_as(c_void_p),pspsz,binsp,int(nBins),mabp)

            shuffledList.append(modArrShuffle)

        meanArr = np.mean(shuffledList,0)
        stdArr = np.std(shuffledList,0)      
        zscoreArr = (modArr - meanArr) / stdArr

    else:
        meanArr, stdArr, zscoreArr = None, None, None

    #   # old z-score implementation 
    #     stdev_modArr = modArr.std()
    #     modArr = (modArr - modArr.mean()) / float(stdev_modArr)
    
    return phaseFreq, ampFreq, modArr, meanArr, stdArr, zscoreArr
  else:
    return None

# variable bandwidth modulation index array
def varModIndArr (raw_signal, sampr, from_t, to_t, phaseCntrFrom, phaseCntrTo, ampCntrFrom, ampCntrTo,\
                 phaseStep=1, ampStep=5, phaseBW=2, nBins=18, ampBWFctr=2):
  '''calculates modulation index array using a variable bandWidth for filtering modulated freq.
  Returns the modulation index array, along with the arrays to plot the x-axis (freq for phase) and
  the y-axis (freq for amp)'''
  phaseFreq = numpy.r_[phaseCntrFrom:phaseCntrTo+1:phaseStep]     #array dimensions:
  ampFreq = numpy.r_[ampCntrFrom:ampCntrTo+1:ampStep]
  modArr = numpy.zeros((ampFreq.size, phaseFreq.size), dtype = float)
  for adx in range(len(ampFreq)):
    amp = ampFreq[adx]
    for pdx in range(len(phaseFreq)):
      phs = phaseFreq[pdx]
      modArr[adx][pdx] = calcMI(raw_signal, sampr, from_t, to_t, phs, amp, phaseBW, phs*ampBWFctr, nBins)
  return phaseFreq, ampFreq, modArr


def plotMI(raw_signal, sampr, from_t, to_t, f_phaseMin, f_phaseMax, f_ampMin, f_ampMax, phaseStep = 1, phase_bandWidth = 2,\
           ampStep = 5, amp_bandWidth = 10, phaseBins = 18, Z_score=False):
    '''will plot a color plot of modulation index with phase freq on x-axis and amp freq on y-axis'''
    phaseFreq, ampFreq, modArr = modIndArr(raw_signal, sampr, from_t, to_t, f_phaseMin, f_phaseMax,\
                                           f_ampMin, f_ampMax, phaseStep, phase_bandWidth,\
                                           ampStep, amp_bandWidth, phaseBins, Z_score)
    if Z_score:
        print('Plotting Z score of MI...')
        title('Z score of MI')
    else:
        print('Plotting MI ...')
        title('Modulation Index')
    contourf(phaseFreq, ampFreq, modArr)
    axis([phaseFreq.min(), phaseFreq.max(), ampFreq.min(), ampFreq.max()])
    colorbar()
    xlabel('freq for phase')
    ylabel('freq for amplitude')
    show()
    return phaseFreq, ampFreq, modArr

def plotArrMI(phaseFreq, ampFreq, modArr):
    '''will take arrays generated by either modIndArr or varModIndArr and plot it'''
    contourf(phaseFreq, ampFreq, modArr)
    axis([phaseFreq.min(), phaseFreq.max(), ampFreq.min(), ampFreq.max()])
    colorbar()
    xlabel('freq for phase')
    ylabel('freq for amplitude')
    show()

def plotArrMIclrBar(phaseFreq, ampFreq, modArr, v_max, v_increm):
    '''will take arrays generated by either modIndArr or varModIndArr and plot it. the color bar values are to be determined.
    v_max is the maximum value on the color bar. v_increm is increments between color gradient within the color bar'''
    v = numpy.arange(0, v_max+v_increm, v_increm) #v_increm added to v_max since stop value is not included in the returned list
    contourf(phaseFreq, ampFreq, modArr, v)
    axis([phaseFreq.min(), phaseFreq.max(), ampFreq.min(), ampFreq.max()])
    colorbar(format='%.5f')
    xlabel('freq for phase')
    ylabel('freq for amplitude')
    show()  

def plotPhAmp(raw_signal, sampr, f_phaseCntr, f_phaseWindow,\
              f_ampCntr, f_ampWindow, nBins = 18):
    '''raw_signal is hoc vector (from which mean is subtracted). function will plot 2 cycles of the phase of f_phaseCntr, filtered
    using band width of f_phaseWindow, plotted against the amplitude of f_ampCntr,
    filtered using band width of f_ampWindow. Better to use the same window size as
    MI plot. Note that 1 second will be remvoved from each end of the signal before calc & plotting phase-amp plot,
    so the raw_signal has to be 2 seconds more than the duration you want to calculate over'''
    half_phWin = f_phaseWindow / float(2)# will filter with band width equal half of bandwidth on either side of central freq
    half_ampWin = f_ampWindow / float(2)
    raw_signal = numpy.array(raw_signal)
    phaseHilb = gethilbnq(raw_signal, sampr, f_phaseCntr-half_phWin, f_phaseCntr+half_phWin)
    phaseFreq = phaseHilb.v[1]
    filt_phase = phaseHilb.v[2]
    ampHilb = gethilbnq(raw_signal, sampr, f_ampCntr-half_ampWin, f_ampCntr+half_ampWin)
    ampFreq = ampHilb.v[0]
    filt_amp = ampHilb.v[2]
    list_Bins = listBins(nBins)
    meanAmpBinsVec = meanAmpBins(phaseFreq, ampFreq, list_Bins)
    MI = modIndex(meanAmpBinsVec)
    #normalize amplitude
    sumMeanAmpBins = meanAmpBinsVec.sum()
    meanAmpBinsVec.div(sumMeanAmpBins)
    h.nqsdel(phaseHilb) # to free memory from nqs
    h.nqsdel(ampHilb) # to free memory from nqs
    #generate vectors for x & y axis 
    rwidth = 360 / float(nBins)
    x_axisBins = h.Vector(nBins*2)#to plot 2 cycles
    x_axisBins.indgen(-180, int(rwidth))
    y_axisVec = h.Vector()
    y_axisVec.append(meanAmpBinsVec)
    y_axisVec.append(meanAmpBinsVec)#to plot 2 cycles
    xlabel('Phase (Deg)')
    ylabel('normalized Mean Amplitude')
    plot(x_axisBins, y_axisVec, label = str(f_phaseCntr)+' / ' + str(f_ampCntr) + ' by ' \
    + '('+str(f_phaseWindow)+') / ('+str(f_ampWindow)+'), MI=' + str(MI))
    axis([x_axisBins.min(), x_axisBins.max(), 0, 0.1])
    xticks(numpy.arange(-180, 541, 90))# to plot 2 cycles
    return filt_phase, filt_amp, meanAmpBinsVec 

#
def checkSize (raw_signal, sampr, from_t, to_t):
    if raw_signal.size() < 2*sampr:
        print("ERRA: Make sure your raw_signal is more than 2 seconds in duration,")
        return False
    if from_t < 1: 
        print("ERRB: Make sure there is at least 1 second before from_t")
        return False
    if (to_t+1)*sampr > raw_signal.size():
        print("ERRC: Make sure there is at least 1 second after to_t")
        return False
    return True

#
def calcMI (raw_signal, sampr, from_t, to_t, phPeak, ampPeak, phBW = 1, ampBW=20, nbins=18):
    '''calculates & returns MI value at a particular frequency pair (phPeak, ampPeak).
    raw_signal is a hoc vector. Duration of raw_signal has to be longer than 2 seconds (since 1 second is removed
    after filtering from each end of the filtered signal). from_t & to_t (in seconds) is the segment from which to calculate
    the coupling.'''
    if checkSize(raw_signal,sampr,from_t,to_t):
        half_phBW = phBW / 2.0
        half_ampBW = ampBW / 2.0
        signal = h.Vector()
        signal.copy(raw_signal, (from_t-1)*sampr, (to_t+1)*sampr-1) # add one second on each side of the signal
        signal = numpy.array(signal)
        last_1s = [signal.size-1-sampr, signal.size-1]
        first_1s = [0, int(sampr-1)]
        # filtering for phase
        phHilb = gethilbnq(signal, sampr, phPeak-half_phBW, phPeak+half_phBW)
        phFreq = phHilb.v[1]
        phFreq.remove(last_1s[0], last_1s[1])
        phFreq.remove(first_1s[0], first_1s[1])
        #filtering for amplitude
        ampHilb = gethilbnq(signal, sampr, ampPeak-half_ampBW, ampPeak+half_ampBW)
        ampFreq = ampHilb.v[0]
        ampFreq.remove(last_1s[0], last_1s[1])
        ampFreq.remove(first_1s[0], first_1s[1])
        list_Bins = listBins(nbins)
        MI = modIndex(meanAmpBins(phFreq, ampFreq, list_Bins))
        h.nqsdel(phHilb) # to free memory from nqs
        h.nqsdel(ampHilb) # to free memory from nqs
        return MI
    else:
        print("calcMI ERROR: Couldn't calculate MI.")
        print('from:', from_t, ' to:', to_t)
        return -1

#      
def MIpeakTime(raw_signal, sampr, phPeak, ampPeak, dur = 1, start_t = 1, phBW = 1, ampBW=20, numPts=100, randPoints = False):
    '''will return a list of MI values, calculated at phPeak & ampPeak, with phase bandWidth phBW
    and amplitude bandWidth ampBW. numPts is number of points to be calculated.  If randPoints = True,
    will calculate MI obtained from random dur. If randPoints = False, will calculate MI obtained from consecutive dur, starting 
    from start_t. dur & start_t: in seconds. raw_signal is a hoc vector, (from which the mean has been subtracted)'''
    points = []
    init = 0
    time0, clock0 = time(), clock()
#    raw_signal = numpy.array(raw_signal)
    if randPoints:
        for i in range(numPts):
            randInit = random.randint(1, int((raw_signal.size()/sampr)-int(dur+1)))
            points.append(calcMI(raw_signal, sampr, randInit, randInit+dur, phPeak, ampPeak, phBW, ampBW))
#            if not numPts%10: gc.collect()
        print('time:', time()-time0, 'clock:', clock()-clock0)
        return points
    else:
        init = start_t
        for i in range(numPts):
            points.append(calcMI(raw_signal, sampr, init, init+dur, phPeak, ampPeak, phBW, ampBW))
            init += dur            
        print('time:', time()-time0, 'clock:', clock()-clock0)
        return points
#      
def avgVarMIarr(raw_signal, sampr, dur, phaseCntrFrom, phaseCntrTo, ampCntrFrom, ampCntrTo,\
                from_t = 1, phaseStep=1, ampStep=5, phaseBW=1, numArr=10, nBins=18):
    '''calculate & returns MI array averaged from numArr MI arrays. returns the averaged MI array and the arrays to plot x & y axis.
raw_signal is a hoc vector. Each MI array will be calculated over time dur, starting at from_t'''
    #generate arrays for plotting & calculation    
    phaseFreq = numpy.r_[phaseCntrFrom:phaseCntrTo+1:phaseStep]
    ampFreq = numpy.r_[ampCntrFrom:ampCntrTo+1:ampStep]
    sumModArr = numpy.zeros((ampFreq.size, phaseFreq.size))
    init = float(from_t) # this is time from which the calculation will start
    for i in range(numArr):
        print('array num:', i)
        myphaseFreq, myampFreq, modArr = varModIndArr(raw_signal, sampr, init, init+dur, phaseCntrFrom, phaseCntrTo, ampCntrFrom, ampCntrTo, phaseStep, ampStep, phaseBW)
        sumModArr += modArr
        init += dur
    avgModArr = sumModArr / float(numArr)
    phaseFreq += (phaseStep / 2.0)
    ampFreq += (ampStep / 2.0)
    return phaseFreq, ampFreq, avgModArr

def avgVarMIarrMulPr(raw_signal, sampr, dur, phaseCntrFrom, phaseCntrTo, ampCntrFrom, ampCntrTo,\
                from_t = 1, phaseStep=1, ampStep=5, phaseBW=1, numArr=10, nBins=18):
    '''calculate & returns MI array averaged from numArr MI arrays. returns the averaged MI array and the arrays to plot x & y axis.
    Each MI array will be calculated over time dur, starting at from_t (in seconds). Same as avgVarMIarr but using multiple python processes to save time.
    Each array will run using a separate python process using multiprocessing module'''
    def varModIndArrPrll(timeFrom, timeTo, out_q):
        '''use the function varModIndArr to calc MIarr for time: timeFrom to timeTo. Outputs the modInd array into modIndOut_q'''
        phaseFreq, ampFreq, MIarr = varModIndArr(raw_signal, sampr, timeFrom, timeTo, phaseCntrFrom, phaseCntrTo, ampCntrFrom, ampCntrTo, phaseStep, ampStep, phaseBW)
        modInd_Out_q.put(MIarr) # I THINK THIS HAS TO BE: out_q.put(MIarr)
    # generate times to calculate each of the different arrays that will be calculated in parallel
    timeList = numpy.arange(from_t, from_t+dur*numArr, dur)
    ##print 'timeList size:', len(timeList)
    # generate arrays that will be used for plotting
    phaseFreq = numpy.r_[phaseCntrFrom:phaseCntrTo+1:phaseStep]
    ampFreq = numpy.r_[ampCntrFrom:ampCntrTo+1:ampStep]
    sumModArr = numpy.zeros((ampFreq.size, phaseFreq.size)) # array of zeros to calculate sum of modIndex arrays
    procList = [] # list of multiple processes
    modInd_Out_q = multiprocessing.Queue()#instantiate the queue that will collect arrays generated by multiple processes
    modArrList = [] # this will collect the modulation arrays generated from multiple processes
    # start multiple parallel processes
    for i in range(numArr):
        print('starting parallel process num:', i)
        p = multiprocessing.Process(name='array num:'+str(numArr), target=varModIndArrPrll, args=(timeList[i],timeList[i]+dur, modInd_Out_q))
        procList.append(p)
        p.start()
    # obtain calculated modulation index arrays from multiple parallel processes
    for i in range(numArr):
        modArrList.append(modInd_Out_q.get())
    # wait till all processes finish
    for p in procList:
        p.join()
    # sum all modulation index arrays
    for myarr in modArrList:
        sumModArr += myarr
    avgModArr = sumModArr / float(numArr) # calculate average array
    phaseFreq += (phaseStep / 2.0) # for plotting
    ampFreq += (ampStep / 2.0)
    return phaseFreq, ampFreq, avgModArr

# get frequencies associated with max CFC in phase,amplitude dimension
def MIPeak (phs,amp,mi):
    i,j = numpy.unravel_index(mi.argmax(), mi.shape)
    return phs[j],amp[i]

