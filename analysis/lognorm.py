from scipy import stats 
import numpy as np
from matplotlib import pyplot as plt

def lognorm (meaninput, stdinput):
	M = float(meaninput) # Geometric mean == median
	s = float(stdinput) # Geometric standard deviation
	mu = np.log10(M) # Mean of log(X)
	sigma = np.log10(s) # Standard deviation of log(X)
	print(mu, sigma)
	shape = sigma # Scipy's shape parameter
	scale = np.power(10, mu) # Scipy's scale parameter
	median = np.power(10, mu)
	mode = np.power(10, mu - sigma**2) # Note that mode depends on both M and s
	mean = np.power(10, mu + (sigma**2/2)) # Note that mean depends on both M and s
	x = np.linspace(0.1, 25, num=400) # values for x-axis
	#xlog = np.logspace(0.1, 25, num=400) # values for x-axis
	xlog = np.logspace(np.log10(0.1), np.log10(25), 400)
	pdf = stats.lognorm.pdf(x, shape, loc=0, scale=scale) # probability distribution

	# sample and fit
	samp = stats.lognorm(shape,loc=0,scale=scale).rvs(size=2000)
	(shape_fit, loc_fit, scale_fit) = stats.lognorm.fit(samp, loc=0)   
	pdf_fit = stats.lognorm.pdf(x, shape_fit, loc=0, scale=scale_fit)

	samp_log = np.log(samp)
	(shape_fit_log, loc_fit_log, scale_fit_log) = stats.lognorm.fit(samp_log, loc=0)   
	pdf_fit_log = stats.lognorm.pdf(x, shape_fit_log, loc=0, scale=scale_fit_log)

	plt.figure(figsize=(12,4.5))
	# Figure on linear scale
	plt.subplot(151)
	plt.plot(x, pdf)
	plt.vlines(mode, 0, pdf.max(), linestyle=':', label='Mode')
	plt.vlines(mean, 0, stats.lognorm.pdf(mean, shape, loc=0, scale=scale), linestyle='--', color='green', label='Mean')
	plt.vlines(median, 0, stats.lognorm.pdf(median, shape, loc=0, scale=scale), color='blue', label='Median')
	plt.ylim(ymin=0)
	plt.xlabel('Radius (microns)')
	plt.title('Linear scale')
	leg=plt.legend()

	# Figure on logarithmic scale
	plt.subplot(152)
	plt.semilogx(x, pdf)
	plt.vlines(mode, 0, pdf.max(), linestyle=':', label='Mode')
	plt.vlines(mean, 0, stats.lognorm.pdf(mean, shape, loc=0, scale=scale), linestyle='--', color='green', label='Mean')
	plt.vlines(median, 0, stats.lognorm.pdf(median, shape, loc=0, scale=scale), color='blue', label='Median')
	plt.ylim(ymin=0)
	plt.xlabel('Radius (microns)')
	plt.title('Logarithmic scale')
	leg=plt.legend()

	# Figure on logarithmic scale 2
	plt.subplot(153)
	weights = np.ones_like(samp)/float(len(samp)) 
	plt.hist(samp, bins=x, histtype='step', normed=True)#, normed=True)#weights=weights)#, normed=True) # )
	plt.plot(x, pdf)
	plt.plot(x, pdf_fit)
	plt.xlabel('Radius (microns)')
	plt.title('Logarithmic scale')
	leg=plt.legend()

	# Figure 
	plt.subplot(154)
	weights = np.ones_like(samp)/float(len(samp)) 
	plt.hist(samp, bins=x, histtype='step', normed=True)#weights=weights)#, normed=True) # )
	plt.xscale('log')
	plt.semilogx(x,pdf)
	plt.semilogx(x, pdf_fit)
	plt.xlabel('Radius (microns)')
	plt.title('Logarithmic scale')
	leg=plt.legend()

	# Figure on logarithmic scale 2
	plt.subplot(155)
	weights = np.ones_like(samp)/float(len(samp)) 
	plt.hist(samp_log, bins=x, histtype='step', normed=True)#weights=weights)#, normed=True) # )
	plt.plot(x, pdf_fit_log)
	# plt.xscale('log')
	# plt.semilogx(x,pdf)
	# plt.semilogx(x, pdf_fit)
	plt.xlabel('Radius (microns)')
	plt.title('Logarithmic scale')
	leg=plt.legend()

	plt.show()

def shapiro_norm():
	# Shapiro-Wilk Test
	from numpy.random import seed
	from numpy.random import randn,rand
	from scipy.stats import shapiro
	# seed the random number generator
	seed(1)
	# generate univariate observations
	data = 5 * rand(100) + 50
	# normality test
	stat, p = shapiro(data)
	print('Statistics=%.3f, p=%.3f' % (stat, p))
	# interpret
	alpha = 0.05
	if p > alpha:
		print('Sample looks Gaussian (fail to reject H0)')
	else:
		print('Sample does not look Gaussian (reject H0)')

# Main code

# lognorm(np.power(10,0.55), np.power(10, 1.5))
# lognorm(6.2,8.1)
shapiro()

'''links:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
https://stackoverflow.com/questions/8747761/scipy-lognormal-distribution-parameters
https://stackoverflow.com/questions/18534562/scipy-lognormal-fitting
http://nbviewer.jupyter.org/url/xweb.geos.ed.ac.uk/~jsteven5/blog/lognormal_distributions.ipynb
https://math.stackexchange.com/questions/497878/how-to-convert-a-histogram-to-a-pdf
http://firsttimeprogrammer.blogspot.com/2015/01/how-to-estimate-probability-density.html
https://stats.stackexchange.com/questions/315714/r-shapiro-wilk-test-for-log-normal-distribution
https://stackoverflow.com/questions/15415455/plotting-probability-density-function-by-sample-with-matplotlib
https://stackoverflow.com/questions/41000668/log-x-axis-for-histogram'''
 
''''prev implementation
        elif graphType == 'histogram':
            
            nmax = 0
            binmax = 0
            for i,data in enumerate(statData):  # fix 
                # if histlogx:
                #     histbins = np.logspace(np.log10(histlogxmin), np.log10(max(data)), bins)
                # else:
                #     histbins = bins

                if histmin: # min value 
                    data = np.array(data)
                    data = data[data>histmin]
                
                if histlogy:
                    data = [np.log10(x) for x in data]

                # if density:
                #     weights = np.ones_like(data)/float(len(data)) 
                # else: 
                #     weights = np.ones_like(data)

                n, binedges,_ = plt.hist(data,  bins=histbins, histtype='step', color=colors[i], linewidth=1.5, normed=density)# weights=weights)
                # plt.hist(data, bins=histbins, alpha=0.25, color=colors[i], linewidth=0, normed=density)# weights=weights) 
                # label = legendLabels[-i-1] if legendLabels else str(include[-i-1])
                # plt.hist([-10], bins=histbins, fc=((colors[i][0], colors[i][1], colors[i][2],0.25)), edgecolor=colors[i], linewidth=1.5, label=label)
                nmax = max(nmax, max(n))
                binmax = max(binmax, binedges[-1])
                if histlogx: 
                    plt.xscale('log')

                if normfit:
                    (shape, loc, scale) = scipy.stats.lognorm.fit(data, loc=0)  
                    mu = np.log(scale)
                    sigma = shape
                    x=np.linspace(0, 25, 400)
                    x=binedges
                    y = scipy.stats.lognorm.pdf(x, shape, loc=0, scale=scale)
                    plt.plot(x, y, '--', color = colors[i], linewidth=2)
                    print binedges, y
                    print shape, scale
                    print mu, sigma
                    if histlogx:
                        plt.xscale('log')

                    # check normality of distribution
                    #W, p = scipy.stats.shapiro(data)
                    #print 'Pop %s rate: mean = %f, std = %f, normality (Shapiro-Wilk test) = %f, p-value = %f' % (include[i], mu, sigma, W, p)


            plt.xlabel(xlabel, fontsize=fontsiz)
            plt.ylabel('Probability of occurrence' if density else 'Frequency', fontsize=fontsiz)
            xmax = binmax
            plt.xlim(histmin, xmax)
            plt.ylim(0, 1.5*nmax if density else np.ceil(1.5*nmax)) #min(n[n>=0]), max(n[n>=0]))
            plt.legend(fontsize=fontsiz)

            plt.show() # REMOVE!

            if xlim: ax.set_xlim(xlim)

            from IPython import embed; embed()
'''