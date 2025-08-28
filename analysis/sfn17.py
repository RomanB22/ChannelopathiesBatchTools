# Code for sfn17 poster

import utils
import json, pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb
import os
plt.style.use('seaborn-whitegrid')

def freqAnalysis2():
    maxAmpsIT2 = [[  3.23660174,  -3.70426414,  -4.32875842, -12.21366434,
        -10.98058524, -10.29860795, -12.2765607 , -11.03269058,
        -14.23800105, -16.48608694],
       [  3.00134733,  -4.73604847,  -5.0894849 , -10.9030432 ,
         -8.82755596,  -7.67688496,  -8.48438478,  -9.96010619,
        -13.14977243, -14.7253491 ],
       [  5.80585479,  -2.35815241,  -4.30033206, -11.07365289,
         -9.39752788, -10.95832612, -11.06284982, -11.9512322 ,
        -16.36935978, -17.90280994],
       [  5.78766085,  -3.79920162,  -5.67654043,  -9.37782993,
         -8.01685029,  -9.17272207,  -7.55809929, -10.06696308,
        -13.62977141, -14.68256828]]

    maxAmpsIT5A = [[ 12.65486133,   5.46694681,   2.41016898,  -0.12690111,   2.73674208,
    8.58878431,  -7.16498416,  -5.04991132,  -7.37467616,  -2.60814664],
 [ 11.43089415,   0.92160424,   3.21658804,   3.12583473,   5.28329177,
    9.27899798,  -5.93434731,  -6.39370089,  -0.71794821,  -2.09033586],
 [  9.46658475,   5.69970561,   3.73730122,   0.03334918,   4.21189979,
    5.98665879,  -9.36594669,  -6.90036945,  -2.94698346,  -2.26403856],
 [ 11.57498741,   2.54466327,   2.99979555,   4.2015383,    4.29711327,
    5.39664214,  -3.23359672,  -5.22851717,  -4.75574678,  -2.56929713]]
    
    maxAmpsPT5B = [[ -0.99527703,  -6.92419375,  -6.92579109, -10.07572835,  -0.99036542,
    9.08125883,  -4.86313418,  -6.52169044, -10.77526453,  -8.8261937 ],
 [ -0.12215353,  -6.72864062,  -5.30155173,  -6.30806402,   2.37106219,
    9.27158496,  -6.39998121,  -7.21532715, -12.73975763,  -5.87216872],
 [  4.28120685,  -6.28550073,  -6.75469468,  -8.98520572,   0.69063433,
    7.2804442,   -4.62986574,  -6.52837586, -10.58707525,  -8.65623012],
 [  3.81638193,  -5.45444757,  -6.56917968,  -4.8505341,    0.73388972,
    6.4647034,   -2.87213217  ,-8.77920895,  -6.61548589, -11.04082724]]

    freqs=[4,8,12,16,20,24,28,32,36,40]
    #freqs=[1]*10

    # maxAmpsIT2 = [[10.0**(x/10.0) for ix,x in enumerate(maxAmpsIT2[i])] for i in range(4)]
    # maxAmpsIT5A = [[10.0**(x/10.0) for ix,x in enumerate(maxAmpsIT5A[i])] for i in range(4)]
    # maxAmpsPT5B = [[10.0**x/10.0 for ix,x in enumerate(maxAmpsPT5B[i])] for i in range(4)]


    # maxAmpsIT2 = [[10*np.log10(x*freqs[ix]) for ix,x in enumerate(maxAmpsIT2[i])] for i in range(4)]
    # maxAmpsIT5A = [[10*np.log10(x*freqs[ix]) for ix,x in enumerate(maxAmpsIT5A[i])] for i in range(4)]
    # maxAmpsPT5B = [[10*np.log10(x*freqs[ix]) for ix,x in enumerate(maxAmpsPT5B[i])] for i in range(4)]

    maxIT2 = np.amax(maxAmpsIT2)

    maxAmpsIT2=map(list, zip(*maxAmpsIT2))
    maxAmpsIT5A=map(list, zip(*maxAmpsIT5A))
    maxAmpsPT5B=map(list, zip(*maxAmpsPT5B))
    

    sb.set_palette('muted')
    plt.style.use('seaborn-whitegrid')

    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 'medium'

    plt.figure(figsize=(6,8))
    plt.plot(maxAmpsIT2, '-o')
    #plt.plot(range(10), [maxIT2/f for f in freqs], 'k:', alpha=0.5)
    plt.xlabel('Stim frequency (Hz)')
    plt.ylabel('Power (dB/Hz) at stim freq')
    plt.xticks(range(len(freqs)), tuple(freqs))
    plt.legend(['start=500ms; low ih', 'start=550ms; low ih', 'start=500ms; high ih', 'start=550ms; high ih'])
    plt.savefig(dataFolder+'/'+batchLabel+'/'+batchLabel+'_maxAmpsIT2.png')

    plt.figure(figsize=(6,8))
    plt.plot(maxAmpsIT5A, '-o')
    plt.xlabel('Stim frequency (Hz)')
    plt.ylabel('Power (dB/Hz) at stim freq')
    plt.xticks(range(len(freqs)), tuple(freqs))
    plt.legend(['start=500ms; low ih', 'start=550ms; low ih', 'start=500ms; high ih', 'start=550ms; high ih'])
    plt.savefig(dataFolder+'/'+batchLabel+'/'+batchLabel+'_maxAmpsIT5A.png')

    plt.figure(figsize=(6,8))
    plt.plot(maxAmpsPT5B, '-o')
    plt.xlabel('Stim frequency (Hz)')
    plt.ylabel('Power (dB/Hz) at stim freq ')
    plt.xticks(range(len(freqs)), tuple(freqs))
    plt.legend(['start=500ms; low ih', 'start=550ms; low ih', 'start=500ms; high ih', 'start=550ms; high ih'])
    plt.savefig(dataFolder+'/'+batchLabel+'/'+batchLabel+'_maxAmpsPT5B.png')

                # return sim,data

freqAnalysis2()