
/* Written by Sam Neymotin (samuel.neymotin@nki.rfmh.org)

this is used for modulation index (cross-frequency coupling) in modindex.py
via the ctypes interface

compilation:
 gcc -Wall -fPIC -c modind.c 
 gcc -shared -o modind.so modind.o

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265358979323846264338327950288419716939937510

/* return the modulation index - 
 meanAmpBins is average amplitude at a given bin
 num is the number of bins
*/
double modindex (double *meanAmpBins,int num) {
  double sumMean = 0.0, sigma=0.0, amp; int i;
  for(i=0;i<num;i++) sumMean += meanAmpBins[i];
  for(i=0;i<num;i++) 
    if ((amp = meanAmpBins[i] / sumMean) > 0.0) sigma -= (amp * log(amp));
  return 1.0 - sigma / log((double)num);
}

/* original python code
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
*/

/* calculates the average amplitude per each phase bin
phaseVec is an array of phases (from the hilbert transform of a bandpass filtered signal)
ampVec is the corresponding amplitude (typically from hilbert transform of bandpassed signal of a different freq)
n is the size of phaseVec,ampVec
listBins are the bins for calculation, numBins is the number
The results are stored in meanAmpBins and the modulation index value is returned.
WARNING: no error checking right now!
*/
double meanampbins (double* phaseVec, double* ampVec, int n, double* listBins, int numBins, double* meanAmpBins) {
  int i,j,bdx;
  double binSize = 2.0 * PI / numBins;
  double *psum = (double*)malloc(sizeof(double)*numBins), *pcnt=(double*)malloc(sizeof(double)*numBins);
  for(i=0;i<numBins;i++) meanAmpBins[i]=psum[i]=pcnt[i]=0.0;
  for(j=0;j<n;j++) {
    bdx = (phaseVec[j]-listBins[0]) / binSize;
    if(bdx<0){printf("bdx<0!\n");bdx=0;}else if(bdx>=numBins){printf("bdx>=numBins:%d,%d\n",bdx,numBins);bdx=numBins-1;}
    psum[bdx] += ampVec[j]; pcnt[bdx]++;
  }
  for(i=0;i<numBins;i++) if(pcnt[i]>0) meanAmpBins[i]=psum[i]/(double)pcnt[i]; 
  free(psum); free(pcnt);
  return modindex(meanAmpBins,numBins);
}

/* original python/NEURON code
#
def meanAmpBins (phaseVec, ampVec, listBins):
    '''input: 2 vectors, each of filtered signal through time, one for phase & other for amplitude; list containing
    the bins of the phase. Returns vector of the values of the mean amp at different bins of phase'''
    numBins = len(listBins)
    meanAmpBins = h.Vector(numBins)
    binSize = 2 * pi / numBins
    meanAmpBins.resize(0)
    indexVec = h.Vector()
    for i in xrange(numBins):
      #index from phaseVec where values lies between listBins[i] and listBins[i]+binSize        
      indexVec.indvwhere(phaseVec, "[)", listBins[i], listBins[i] + binSize)
      amplitude = ampVec.ind(indexVec)#vector of values from ampVec corresponding to indices from phaseVec
      if amplitude.size()>0:
        meanAmp = amplitude.mean()
        meanAmpBins.append(meanAmp)
      else:
        meanAmpBins.append(0.0)
    return meanAmpBins

 */
