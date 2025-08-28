# Extracting spike trains from crcns ssc-3 dataset

from scipy.io import loadmat 
import matplotlib.pyplot as plt
import json

requiredCells = 1000
duration = 5000.0 # ms
minRate = 0.1
maxRate = 50
offset = 500
numDatasets = 25
extracted = 0
dataset = 1
allSpks = []
rates = []

while extracted < requiredCells and dataset <=numDatasets:
	# load new dataset
	data = loadmat('data/DataSet%d.mat'%(dataset), struct_as_record=False, squeeze_me=True) 
	data = data['data']
	numCells = data.nNeuronsç
	cells = data.spikes
	icell = 0
	minSpkt = max([min(cell) for cell in cells])
	while icell < numCells and extracted < requiredCells:
		cell = cells[icell]
		spks = [spk-minSpkt+offset for spk in cell if minSpkt < spk < minSpkt+duration]
		rate = len(spks)/duration*1000.0
		if rate >= minRate and rate<=maxRate:
			print extracted, dataset, icell, rate
			rates.append(rate)
			allSpks.append(spks)
			extracted += 1
		icell += 1
	dataset += 1 
with open('ssc-3_lowrate2_spikes.json', 'w') as f: json.dump(allSpks, f)
plt.hist(rates,20)
plt.show()


