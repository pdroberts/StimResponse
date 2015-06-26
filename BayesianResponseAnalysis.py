#!/usr/bin/python
# BayesianResponseAnalysis.py by Patrick D Roberts (2014)
# Analysis of spike data for responses to auditory stimuli using Bayesian analysis

import numpy as np
import scipy.stats as stats
import pandas as pd
import pymc as pm  # For Bayesian analysis
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns           # Public graphing commands
import matplotlib.cm as cm     # Color map for STH (gray), etc
import cPickle     # For saving and loading data structures from disk
from StimResponse import VocalEphysData   # Custom data analysis commands
#=======================================
class BayesianResponseAnalysis:    
	"""Class for performing a Bayesian analysis of spike responses to stimuli.
	:param initDirectoryPath: Path to directory for saving and reading files
	:type initDirectoryPath: str
	"""
	def __init__(self, initDirectoryPath):
		self.dirPath = initDirectoryPath
		self.spike_histo = np.array([])
		self.lambda_1_samples = np.array([])
		self.lambda_2_samples = np.array([])
		self.tau_samples = np.array([])
		self.tau2_samples = np.array([])
		self.sampleStats = np.array([])
		self.resProb = np.array([])
		self.expected_spikes_per_bin = np.array([])
   
	def ParamStats(self, samples ):
		""" Calculates and returns 1st-4th order statistics for data in samples.        
		:param samples: Array of samples of a parameter following MCMC
		:type samples: numpy array
		:returns: numpy array with statistics of samples (mean, std, skew, kurtosis)
		"""		
		m = np.mean(samples)
		s = np.std(samples)
		sk = stats.skew(samples)
		k = stats.kurtosis(samples)
		return np.array([m, s, sk, k])

	def BayesSpikeResponse(self, spikeData, duration=250, verbose=False, priors=[] ):
		""" Runs Bayesian analysis on spikeData to determine whether the response is significant and calculates effect size.        
		:param spikeData: DataFrame containing the spike time data from a series of stimulus presentations
		:type spikeData: pandas DataFrame
		:param duration: Duration of recording window
		:type duration: float
		:param verbose: Flag for printing progress and plotting results
		:type verbose: boolean
		:param priors: Option to introduce custom priors for tau and tau2. Used for multilevel Bayesian analysis 
		:type priors: list of numpy arrays
		:returns: If verbose flag is True, returns handle to figure of response analysis
		"""		
		self.BayesResponse4(spikeData, duration, p_bar=verbose, priors=priors)
		self.ResponseProbability(duration)
		self.ExpectedSpikesPerBin(spikeData)
		if verbose: 
			return self.PlotResponseEst(self.spike_histo, self.lambda_1_samples, self.lambda_2_samples, self.tau_samples, self.tau2_samples, verbose) 

	def BayesResponse4(self, spikeTrains, duration=250, p_bar=False, priors=[]):  # Bayesian estimation of response with 4 linear parameters 
		""" Performs 4-parameter, piecewise constant Bayesian analysis on spikeData. 
			Bayesian model parameters: 
				 lambda_1 = rate of response
				 lambda_2 = rate of background activity
				 tau = onset time of response 
				 tau2 = duration of response      
		:param spikeTrains: DataFrame containing the spike time data from a series of stimulus presentations
		:type spikeTrains: pandas DataFrame
		:param duration: Duration of recording window
		:type duration: float
		:param p_bar: Flag for printing progress
		:type p_bar: boolean
		:param priors: Option to introduce custom priors for tau and tau2. Used for multilevel Bayesian analysis 
		:type priors: list of 2 numpy arrays
		"""		
		if len(spikeTrains.shape)>1:
			spike_histo = np.histogram(spikeTrains.stack(), bins=duration, range=(0,duration)) 
		else:
			spike_histo = np.histogram(spikeTrains.dropna(), bins=duration, range=(0,duration))
		spike_data = spike_histo[0]
		n_spike_data = len(spike_data)
		rate = spike_data.mean()  # spike_data is the variable that holds our spike counts
		alpha = 1.0 / rate
		np.random.seed(123456)
		if len(priors)==0:
			lambda_1 = pm.Uniform("lambda_1", 0, 2*rate)
			lambda_2 = pm.Uniform("lambda_2", 0, max(spike_data))
			tau = pm.Uniform("tau", 0, 60)
			tau2 = pm.Uniform("tau2", 0, 100)
		else:
			lambda_1 = pm.Uniform("lambda_1", 0, 2*rate)
			lambda_2 = pm.Uniform("lambda_2", 0, max(spike_data))
			tau = priors[0]
			tau2 = priors[1]
		@pm.deterministic
		def lambda_(tau=tau, tau2=tau2, lambda_1=lambda_1, lambda_2=lambda_2):
			out = lambda_1*np.ones(n_spike_data) # lambda_1 is spontaneous spike rate
			out[tau:tau+tau2] = lambda_2  # lambda after (and including) tau is lambda2
			return out    
		observation = pm.Poisson("obs", lambda_, value=spike_data, observed=True)
		model = pm.Model([observation, lambda_1, lambda_2, tau, tau2])
		mcmc = pm.MCMC(model)
# 		mcmc.sample(40000, 10000, 1, progress_bar=p_bar)
		mcmc.sample(20000, 10000, 1, progress_bar=p_bar)
		self.lambda_1_samples = mcmc.trace('lambda_1')[:]
		self.lambda_2_samples = mcmc.trace('lambda_2')[:]
		self.tau_samples = mcmc.trace('tau')[:]
		self.tau2_samples = mcmc.trace('tau2')[:]
		self.spike_histo = spike_data

	def ExpectedSpikesPerBin(self, spike_data):
		""" Calculates piecewise constant firing rate based on parameter posterior calculated by BayesResponse4().        
		:param spikeTrains: DataFrame containing the spike time data from a series of stimulus presentations
		:type spikeTrains: pandas DataFrame
		:returns: numpy array with statistics of samples (mean, std, skew, kurtosis)
		"""		
		spike_data = self.spike_histo
		lambda_1_samples = self.lambda_1_samples
		lambda_2_samples = self.lambda_2_samples
		tau_samples = self.tau_samples
		tau2_samples = self.tau2_samples
		n_spike_data = len(spike_data)
		N = tau_samples.shape[0]
		self.expected_spikes_per_bin = np.zeros(n_spike_data)
		for b in range(0, n_spike_data): 
			ix = tau_samples < b 
			ix2 = b < tau_samples + tau2_samples 
			self.expected_spikes_per_bin[b] = (lambda_1_samples[~(ix&ix2)].sum() 
											 + lambda_2_samples[ix&ix2].sum())/N

	def PlotResponseEst(self, spike_data, lambda_1_samples, lambda_2_samples, tau_samples, tau2_samples, duration=250, verbose=False):
		""" Plots results of Bayesian analysis on spike_data.        
		:param spike_data: DataFrame containing the spike time data from a series of stimulus presentations
		:type spike_data: pandas DataFrame
		:param lambda_1_samples: Array of samples for the lambda_1 parameter following MCMC
		:type lambda_1_samples: numpy array
		:param lambda_2_samples: Array of samples for the lambda_2 parameter following MCMC
		:type lambda_2_samples: numpy array
		:param tau_samples: Array of samples for the tau parameter following MCMC
		:type tau_samples: numpy array
		:param tau2_samples: Array of samples for the tau2 parameter following MCMC
		:type tau2_samples: numpy array
		:param duration: Duration of recording window
		:type duration: float
		:param verbose: Flag for printing progress and plotting results
		:type verbose: boolean
		:returns: If verbose flag is True, returns handle to figure of response analysis
		"""		
		n_spike_data = len(spike_data)
		fig = plt.figure(figsize=(10,5))
		lambda_1_stats = self.ParamStats( lambda_1_samples )
		lambda_2_stats = self.ParamStats( lambda_2_samples )
		tau_stats = self.ParamStats( tau_samples )
		print "\n"
		print 'resProb, resMag, resMag_MLE, effectSize, effectSize_MLE, spontRate, spontRateSTD, resLatency, resLatencySTD, resDuration ='
		print self.resProb
		print "Response magnitide =", self.resProb[2], "\pm", lambda_1_stats[0]+lambda_2_stats[0]
		print "Response latency =", self.resProb[7], "\pm", self.resProb[8]

		gs = gridspec.GridSpec(2, 2)
		ax1 = plt.subplot(gs[0, :])
		ax2 = plt.subplot(gs[1,0])
		ax3 = plt.subplot(gs[1,1])
	
		self.ExpectedSpikesPerBin(spike_data)
		ax1.plot(range(n_spike_data), self.expected_spikes_per_bin, lw=4, color="#E24A33",
				 label="expected number of spikes per bin")
		ax1.set_xlim(0, n_spike_data)
		ax1.set_ylabel("Expected spike count")
		ax1.bar(np.arange(len(spike_data)), spike_data, color="#348ABD", alpha=0.65,
				label="observed spikes per bin")
		ax1.legend(loc="upper right");
	
		ax2.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
				 label="posterior of $\lambda_1$", color="#A60628", normed=True)
		ax2.set_xlabel("$\lambda$ value")
		ax2.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
			 label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
		ax2.legend(loc="upper right")
	
		tau1_hist = ax3.hist(tau_samples, histtype='stepfilled', bins=30, alpha=0.85,
				 label=r"posterior of $\tau_1$", color="k", normed=True)
		tau2_hist = ax3.hist(tau2_samples, histtype='stepfilled', bins=30, alpha=0.85,
				 label=r"posterior of $\tau_2$", color="#467821", normed=True)
		ax3.set_ylim([0, np.max((np.max(tau1_hist[0]), np.max(tau2_hist[0])))])
# 		ax3.set_xlim([0, duration])
		ax3.legend(loc="upper right")
		ax3.set_xlabel(r"$\tau$ (ms)")
	#     ax3.set_ylabel("probability");    
		return fig

	def ResponseProbability(self, binNum=250):	
		""" Calculates Hellinger distance, responses probability, and effect size based on parameter posterior calculated by BayesResponse4().        
		:param binNum: Number of bins in spike histogram
		:type binNum: int
		"""		
		lambda_1_samples = self.lambda_1_samples
		lambda_2_samples = self.lambda_2_samples
		tau_samples = self.tau_samples
		tau2_samples = self.tau2_samples
		maxVal = max((max(lambda_1_samples),max(lambda_2_samples)))
		histo1, binEdges1 = np.histogram(lambda_1_samples, binNum, (0,maxVal))
		histo2, binEdges2 = np.histogram(lambda_2_samples, binNum, (0,maxVal))
		histo1N = histo1/float(histo1.sum())
		histo2N = histo2/float(histo2.sum())
		intersect = np.array(histo1N)
		bhat = np.array(histo1N)
		hel = np.array(histo1N)
		for b in range(len(intersect)):
			intersect[b] = min((histo1N[b], histo2N[b]))
			bhat[b] = np.sqrt(histo1N[b]*histo2N[b]) # Bhattacharyya coefficient sum(sqrt(histo1N_n*histo2N_n))
			hel[b] =  (np.sqrt(histo1N[b])-np.sqrt(histo2N[b]))**2  # Hellinger distance [Costa13], Eq.16
		# print 'Hellinger distance =', np.sqrt(hel.sum()/2)
		distribOverlap = 1-intersect.sum()
		rho = bhat.sum()
		HD = np.sqrt(1 - rho) #Hellinger distance
		maxIdx1 = np.where(histo1==max(histo1))  # find the index of the maximum 
		MLE1 = binEdges1[maxIdx1]  # bin of the maximum
		maxIdx2 = np.where(histo2==max(histo2))  # find the index of the maximum 
		MLE2 = binEdges2[maxIdx2]  # bin of the maximum
		resMagMLE = MLE2-MLE1
		lambda_1_stats = self.ParamStats( lambda_1_samples )
		lambda_2_stats = self.ParamStats( lambda_2_samples )
		tau_stats = self.ParamStats( tau_samples )
		tau2_stats = self.ParamStats( tau2_samples )
		self.sampleStats = np.array((lambda_1_stats, lambda_2_stats, tau_stats, tau2_stats))
		resMag = lambda_2_stats[0]-lambda_1_stats[0]
		lambda_diff = lambda_2_samples - np.flipud(lambda_1_samples)
		effectSize = np.mean(lambda_diff)/np.std(lambda_diff)
		effectSize_MLE = (MLE2-MLE1)/np.std(lambda_diff)
		spontRate = lambda_1_stats[0]
		spontRateSTD = lambda_1_stats[1]
		tau_histo = np.histogram(tau_samples, bins=30, normed=True) #generate a histogram of the response onset times (tau) 
		maxIdx = np.where(tau_histo[0]==max(tau_histo[0]))  # find the index of the maximum 
		latencies = tau_histo[1][maxIdx]  # the time bin of the maxima
		if latencies[0] > 10:
			resLatency = latencies[0] # capture the first maximum of the latency sample distribution
		else: resLatency = tau_stats[0]
		resLatencySTD = tau_stats[1]
		resDuration = tau2_stats[0]
		self.resProb = HD, resMag, resMagMLE[0], effectSize, effectSize_MLE[0], spontRate, spontRateSTD, resLatency, resLatencySTD, resDuration
		
	def CF_ResponseLoop(self, cfResponseData, paramSet=[], verbose=False, filePath=[]):
		""" DEPRECIATED - Automates analysis for a dictionary of spike time data for multiple cells.        
		:param cfResponseData: Dictionary of DataFrames containing the spike time data from series of stimulus presentations
		:type cfResponseData: dict of pandas DataFrame objects
		:param paramSet: DataFrame containing stimulus parameters
		:type paramSet: pandas DataFrame
		:param verbose: Flag for printing progress and plotting results
		:type verbose: boolean
		:param filePath: Path to directory where results will be saved
		:type filePath: str
		:returns: pandas DataFrame containing response analysis results
		"""		
		cfResponse = {}
		cfResponseStats = {}
		for u in cfResponseData.keys():
			spikeTrains = cfResponseData[u]
			if len(paramSet)>0: 
				stimParams = paramSet['TuningCurve'].ix[u]
				duration=stimParams['recDur']
			else:
				duration=250
			if duration==0: duration=250
			self.BayesSpikeResponse(spikeTrains, duration, verbose )
			cfResponse[u] = self.resProb
			cfResponseStats[u] = self.sampleStats
			if len(filePath)>0:
				cPickle.dump(cfResponse, open(self.dirPath + filePath + 'cfResponse.p', 'wb'))
				cPickle.dump(cfResponseStats, open(self.dirPath + filePath + 'cfResponseStats.p', 'wb'))
		return cfResponse, cfResponseStats

	def VocalResponseLoop(self, vocalSpikes, duration=250, verbose=False, filePath=[]):
		""" Automates analysis for a dictionary of spike time data for multiple vocalizations tests.        
		:param vocalSpikes: Dictionary of DataFrames containing the spike time data from series of stimulus presentations
		:type vocalSpikes: dict of pandas DataFrame objects
		:param duration: Duration of recording window
		:type duration: float
		:param verbose: Flag for printing progress and plotting results
		:type verbose: boolean
		:param filePath: Path to directory where results will be saved
		:type filePath: str
		:returns: pandas DataFrame containing response analysis results
		"""		
		vocalResponse = {}
		vocalResponseStats = {}
		for v in vocalSpikes.keys():
			spikeTimes = vocalSpikes[v].dropna()
			if verbose: v
			if len(spikeTimes)>0:
				self.BayesSpikeResponse(spikeTimes, duration, verbose)
				vocalResponse[v] = self.resProb
				vocalResponseStats[v] = self.sampleStats
				if verbose and len(filePath)>0: fig.savefig(self.dirPath + filePath + '/vocalResponses/'+ vSD_key+'_'+ v+'.png', bbox_inches='tight')
		if len(filePath)>0:
			cPickle.dump(vocalResponse, open(self.dirPath + filePath + 'vocalResponse.p', 'wb'))
			cPickle.dump(vocalResponseStats, open(self.dirPath + filePath + 'vocalResponseStats.p', 'wb'))
		return vocalResponse, vocalResponseStats
            
	def SpecTempResponseLoop(self, stRaster, duration=250, verbose=False, filePath=[]):
		""" Automates analysis for a dictionary of spike time data for multiple tone tests.        
		:param stRaster: Dictionary of DataFrames containing the spike time data from series of stimulus presentations
		:type stRaster: dict of pandas DataFrame objects
		:param duration: Duration of recording window
		:type duration: float
		:param verbose: Flag for printing progress and plotting results
		:type verbose: boolean
		:param filePath: Path to directory where results will be saved
		:type filePath: str
		:returns: pandas DataFrame containing response analysis results
		"""		
		ved = VocalEphysData.VocalEphysData(self.dirPath)
		stHistos = ved.Raster2Histo(stRaster)
		stH_key = stHistos.keys()
		freqTuningHisto = stHistos[stH_key]
		orderedKeys, freqs, attns = ved.GetFreqsAttns(freqTuningHisto)
		stResponseAttn = {}
		stResponseProbAttn = {}
		for a in range(len(attns)):
			if len(attns)>1: attn = attns[a]
			else: attn = attns
			stResponse = np.ndarray(shape=(duration,))
			stRespProb = np.ndarray(shape=(10,))
			for freq in range(len(orderedKeys[a][:])):
				if verbose: print '===  ', stH_key, orderedKeys[a][freq], '  ==='
				spikeTrains = stRaster[orderedKeys[a][freq]]
				self.BayesSpikeResponse(spikeTrains, duration, verbose )
				stResponse = np.vstack([stResponse, self.expected_spikes_per_bin])
				stRespProb = np.vstack([stRespProb, np.array(self.resProb)])
			stResponseAttn[str(int(attn))+'dB'] = stResponse
			stResponseProbAttn[str(int(attn))+'dB'] = stRespProb
		f = map(str,freqs.astype(int))
		f[:0] = ['']
		stResponse = pd.Panel(stResponseAttn, major_axis=f)
		stResponseProb = pd.Panel(stResponseProbAttn, major_axis=f)
		if len(filePath)>0:
			cPickle.dump(stResponse, open(self.dirPath + filePath, 'wb'))
			cPickle.dump(stResponseProb, open(self.dirPath + filePath, 'wb'))
		return stResponse, stResponseProb
       
	def BBNResponseLoop(self, stRaster, duration=250, verbose=False, filePath=[]):
		""" Automates analysis for a dictionary of spike time data for multiple broadband noise tests.        
		:param stRaster: Dictionary of DataFrames containing the spike time data from series of stimulus presentations
		:type stRaster: dict of pandas DataFrame objects
		:param duration: Duration of recording window
		:type duration: float
		:param verbose: Flag for printing progress and plotting results
		:type verbose: boolean
		:param filePath: Path to directory where results will be saved
		:type filePath: str
		:returns: pandas DataFrame containing response analysis results
		"""		
		ved = VocalEphysData.VocalEphysData(self.dirPath)
		stHistos = ved.Raster2Histo(stRaster)
		stH_key = stHistos.keys()
		stResponseDict = {}
		stResponseProbDict = {}
		freqTuningHisto = stHistos[stH_key]
		orderedKeys, attns = ved.GetAttns(freqTuningHisto)
		stResponse = np.ndarray(shape=(duration,))
		stRespProb = np.ndarray(shape=(10,))
		for a in range(len(orderedKeys)):
			spikeTrains = stRaster[orderedKeys[a]]
			self.BayesSpikeResponse(spikeTrains, duration, verbose )
			stResponse = np.vstack([stResponse, self.expected_spikes_per_bin])
			stRespProb = np.vstack([stRespProb, np.array(self.resProb)])
		stResponseDF = pd.DataFrame(stResponse[1:], index=map(str,attns.astype(int)) )
		stResponseProbDF = pd.DataFrame(stRespProb[1:], index=map(str,attns.astype(int)) )
		if len(filePath)>0:
			cPickle.dump(stResponseDF, open(self.dirPath + filePath, 'wb'))
# 			cPickle.dump(stResponseProbDF, open(self.dirPath + filePath, 'wb'))
		return stResponseDF, stResponseProbDF

	def BBN_threshold(self, bbnResponseProb, unit, respSignif=0.95):
		""" Finds BBN response threshold.        
		:param bbnResponseProb: DataFrames results of Bayesian response analysis for multiple BBN stimulus intensities
		:type bbnResponseProb: pandas DataFrame 
		:param unit: Unique identifier for cell
		:type unit: str
		:param respSignif: Significance level for threshold determination. 
		:type respSignif: float
		:returns: float: Minimal stimulus intensity for significant BBN response
		"""		
		measure = 0  # responsProb
		from scipy.interpolate import interp1d
		hd = np.array(bbnResponseProb.loc[:,measure].fillna(0))
		att = np.array(bbnResponseProb.loc[:,measure].fillna(0).index).astype(np.float)
		responseCurve = interp1d(att, hd)
		if min(hd) == max(hd):
			return max(att)
		else:
			return min(np.where(responseCurve(np.arange(min(att), max(att)))>respSignif)[0])+min(att)
            
	def PlotFrequencyTuningCurves(self, stResponseProb, measure, unit=[], filePath=[]):
		""" Plots measure for multiple frequencies, with a trace fro each tone intensity.        
		:param stResponseProb: DataFrames results of Bayesian response analysis for multiple tone stimulus intensities
		:type stResponseProb: pandas DataFrame 
		:param measure: Bayesian response analysis measure ['resProb', 'vocalResMag', 'vocalResMag_MLE', 'effectSize', 'effectSize_MLE', 'spontRate', 'spontRateSTD', 'responseLatency', 'responseLatencySTD', 'responseDuration']
		:type measure: int [0-9]
		:param unit: Unique identifier for cell
		:type unit: str
		:param filePath: Path to directory where results will be saved
		:type filePath: str
		:returns: Handle to plot
		"""		
		measureName = ['resProb', 'vocalResMag', 'vocalResMag_MLE', 'effectSize', 'effectSize_MLE', 'spontRate', 'spontRateSTD', 'responseLatency', 'responseLatencySTD', 'responseDuration']
		tuningData = stResponseProb
		sns.set_palette(sns.color_palette("bright", 8))
		attn = stResponseProb.keys()[0]
		firstFreq = stResponseProb[attn].index.tolist()[1]
		ax = stResponseProb.loc[:,firstFreq:,measure].fillna(0).plot(figsize=(6,4))
		sns.despine()
		sns.set_style("white")
		plt.grid(False)
		plt.title(unit, fontsize=14)
		plt.xlabel('Frequency (kHz)', fontsize=12)
		plt.ylabel(measureName[measure], fontsize=12)
		plt.tick_params(axis='both', which='major', labelsize=14)
		if len(filePath)>0:
			plt.savefig(self.dirPath + filePath + 'freqTuning_'+measureName[measure]+'_'+unit+'.pdf')        
			plt.close()
		else: plt.show()
		return ax
	
	def PlotFrequencyResponseArea(self, stResponseProb, measure, unit=[], filePath=[]):
		""" Plots measure for multiple frequencies and intensities as a contour plot.        
		:param stResponseProb: DataFrames results of Bayesian response analysis for multiple tone stimulus intensities
		:type stResponseProb: pandas DataFrame 
		:param measure: Bayesian response analysis measure ['resProb', 'vocalResMag', 'vocalResMag_MLE', 'effectSize', 'effectSize_MLE', 'spontRate', 'spontRateSTD', 'responseLatency', 'responseLatencySTD', 'responseDuration']
		:type measure: int [0-9]
		:param unit: Unique identifier for cell
		:type unit: str
		:param filePath: Path to directory where results will be saved
		:type filePath: str
		:returns: Handle to plot
		"""		
		measureName = ['resProb', 'vocalResMag', 'vocalResMag_MLE', 'effectSize', 'effectSize_MLE', 'spontRate', 'spontRateSTD', 'responseLatency', 'responseLatencySTD', 'responseDuration']
		if len(stResponseProb) >1:
			if measure==0: colorRange = (0,1.1)       #'resProb'
			elif measure==1: colorRange = (-10,10.1)    #'vocalResMag'
			elif measure==2: colorRange = (-3,3)     #'vocalResMag_MLE'
			elif measure==3: colorRange = (-10,10.1)     #'effectSize'
			elif measure==4: colorRange = (-10,10.1)     #'effectSize_MLE'
			elif measure==5: colorRange = (0,0.5)    #'spontRate'
			elif measure==6: colorRange = (0,0.1)    #'spontRateSTD'
			elif measure==7: colorRange = (0,60)    #'responseLatency'
			elif measure==8: colorRange = (0,30)    #'responseLatencySTD'
			elif measure==9: colorRange = (0,100)    #'responseDuration'
			tuningCurveDF = stResponseProb.loc[:,:,measure]
			F = np.array(tuningCurveDF.index.tolist())[1:].astype(np.float)
			A = tuningCurveDF.keys().tolist()
			A = np.array([a.replace('dB', '') for a in A]).astype(np.float)
			levelRange = np.arange(colorRange[0], colorRange[1], (colorRange[1]-colorRange[0])/float(25*(colorRange[1]-colorRange[0]))) 
			firstFreq = str(int(np.min(F)))
			sns.set_context(rc={"figure.figsize": (7, 4)})
			ax = plt.contourf(F, A, tuningCurveDF.loc[firstFreq:,:].fillna(0).T, vmin=colorRange[0], vmax=colorRange[1], levels=levelRange, cmap = cm.bwr )
			plt.colorbar()
			plt.title(unit +', '+ measureName[measure], fontsize=14)
			plt.xlabel('Frequency (kHz)', fontsize=14)
			plt.ylabel('SPL (dB)', fontsize=14)
# 			plt.gca().invert_yaxis()
			if len(filePath)>0:
				plt.savefig(self.dirPath + filePath + 'freqResponse_'+measureName[measure]+'_'+unit+'.png')        
				plt.close()
			else: plt.show()
		else: self.PlotFrequencyTuningCurves(stResponseProb, unit, measure, save)
		return ax
		
	def PlotSTResponseEst(self, stResponseDF, label, duration=250, firstFreq=1):
		""" Plots response rate estimate for multiple frequencies and intensities as a contour plot.        
		:param stResponseDF: DataFrames results of Bayesian response analysis for multiple tone stimulus intensities
		:type stResponseDF: pandas DataFrame 
		:param label: Figure name
		:type label: str
		:param duration: Duration of recording window
		:type duration: float
		:param firstFreq: Set to skip first (spurious) entry 
		:type firstFreq: int
		:returns: Handle to plot
		"""		
		stResponseE = np.array(stResponseDF)
		freqs = np.array(stResponseDF.index.tolist())[1:].astype(np.float)
		sns.set_context(rc={"figure.figsize": (8, 4)})
		maxRes = np.max(abs(stResponseE[firstFreq:,:]))
		spontRate = np.average(stResponseE[firstFreq:,-1])
		ax = plt.imshow(stResponseE[firstFreq:,:], vmax=maxRes+spontRate, vmin=-maxRes+spontRate, extent=[0,duration,min(freqs),max(freqs)], aspect='auto', interpolation='nearest', origin='lower', cmap = cm.bwr)
		sns.despine()
		plt.grid(False)
		plt.title(label)
		plt.xlabel('Time (ms)')
		plt.ylabel('Frequency (kHz)')
		plt.colorbar()
		return ax

	def PlotBBNResponseCurve(self, bbnResponseProb, measure, unit=[], filePath=[]):
		""" Plots measure for multiple frequencies and intensities an a contour plot.        
		:param stResponseProb: DataFrames results of Bayesian response analysis for multiple tone stimulus intensities
		:type stResponseProb: pandas DataFrame 
		:param measure: Bayesian response analysis measure ['resProb', 'vocalResMag', 'vocalResMag_MLE', 'effectSize', 'effectSize_MLE', 'spontRate', 'spontRateSTD', 'responseLatency', 'responseLatencySTD', 'responseDuration']
		:type measure: integer [0-9]
		:param unit: Unique identifier for cell
		:type unit: str
		:param filePath: Path to directory where results will be saved
		:type filePath: str
		:returns: Handle to plot
		"""		
		measureName = ['resProb', 'vocalResMag', 'vocalResMag_MLE', 'effectSize', 'effectSize_MLE', 'spontRate', 'spontRateSTD', 'responseLatency', 'responseLatencySTD', 'responseDuration']
		tuningData = bbnResponseProb
		sns.set_palette(sns.color_palette("bright", 8))
		attn = bbnResponseProb.keys()[0]
		sns.set_context(rc={"figure.figsize": (5, 3)})
		ax = bbnResponseProb.loc[:,measure].fillna(0).plot(figsize=(6,4))
		sns.despine()
		sns.set_style("white")
		plt.grid(False)
		plt.title(unit, fontsize=14)
		plt.xlabel('SPL (dB)', fontsize=12)
		plt.ylabel(measureName[measure], fontsize=12)
		plt.ylim(0.5,1.0)
# 		plt.gca().invert_xaxis()
		if len(filePath)>0:
			plt.savefig(dirPath + '/TuningCurves/'+ 'freqTuning_'+measureName[measure]+'_'+unit+'.png')        
			plt.close()
		else: plt.show()
		return ax
	
	def SurfPlotFrequencyTuningCurves(self, stResponseProb, measure, unit=[], firstFreq=1):
		""" Plots measure for multiple frequencies and intensities as a 3D plot.        
		:param stResponseProb: DataFrames results of Bayesian response analysis for multiple tone stimulus intensities
		:type stResponseProb: pandas DataFrame 
		:param measure: Bayesian response analysis measure ['resProb', 'vocalResMag', 'vocalResMag_MLE', 'effectSize', 'effectSize_MLE', 'spontRate', 'spontRateSTD', 'responseLatency', 'responseLatencySTD', 'responseDuration']
		:type measure: int [0-9]
		:param unit: Unique identifier for cell
		:type unit: str
		:param firstFreq: Set to skip first (spurious) entry 
		:type firstFreq: int
		:returns: Handle to plot
		"""		
		measureName = ['resProb', 'vocalResMag', 'vocalResMag_MLE', 'effectSize', 'effectSize_MLE', 'spontRate', 'spontRateSTD', 'responseLatency', 'responseLatencySTD', 'responseDuration']
		from mpl_toolkits.mplot3d import Axes3D
		from matplotlib.ticker import LinearLocator, FormatStrFormatter
		import matplotlib.cm as cm     # Color map for surface (coolwarm), etc
		firstF = firstFreq
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		tuningCurveDF = stResponseProb.loc[:,:,measure]
		X = np.array(tuningCurveDF.index.tolist())[firstF:].astype(np.float)
		A = tuningCurveDF.keys().tolist()
		Y = np.array([a.replace('dB', '') for a in A]).astype(np.float)
		X, Y = np.meshgrid(X, Y)
		firstFreq = str(int(np.min(X)))
		Z = tuningCurveDF.loc[firstFreq:,:].fillna(0).T
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap = cm.bwr, linewidth=0, antialiased=False)
		label = unit + ',  ' + measureName[measure]
		ax.set_title(label)
		ax.set_xlabel('Frequency (kHz)') 
		ax.set_ylabel('Attenuation (dB)') 
		# ax.set_zlim(-1.01, 1.01)
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
		ax.view_init(elev=30., azim=300)
		cb = fig.colorbar(surf, shrink=0.5, aspect=5)
		cb.set_label(measureName[measure])
		return ax

	def GenerateSpikes(self, numSpikes, variance, numCycles=20):    # numSpikes <= numCycles
		""" Generate random spike timing data for testing significance scale with controlled statistics.        
		:param numSpikes: Number of spike per response
		:type numSpikes: int 
		:param variance: variability of response
		:type variance: float
		:param numCycles: Number of presentations
		:type numCycles: int
		:returns: pandas DataFrame with spike times for each presentation
		"""		
		spikes = []
		for st in np.random.permutation(range(numCycles)):
			spiketime = (20.+ np.random.exponential(scale=variance))%250    # conincident spikes: scale=0.1
			if st<numSpikes: s = pd.Series([spiketime])
			else:            s = pd.Series([ np.nan ])
			spikes.append(s)
		spikeData = pd.DataFrame(spikes).T
		return spikeData

  