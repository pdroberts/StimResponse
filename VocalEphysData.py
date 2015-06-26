#!/usr/bin/python
# VocalEphysData.py by Patrick D Roberts (2014-2015)
# Import Batlab & Sparkle electrophysiology data into python using parsing scripts from Sparkle (by Amy Boyle, 2015)

import pymatbridge as pymat
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
import pandas as pd
import cPickle     # For saving and loading data structures 
import matplotlib.pyplot as plt
import seaborn as sns           # Public graphing commands for prettier graphs
import matplotlib.cm as cm     # Color map for STH (gray), etc
from pandas import ExcelWriter
import sparkle.data.open as op
import sparkle.tools.spikestats as spst
#=======================================
class VocalEphysData:    
	"""Class for loading and visualizing data from electrophysiology data files 
		Two file formats are supported: Batlab & Sparkle

	:param initDirectoryPath: Path to directory containing metadata on files
	:type initDirectoryPath: str
	:param initBatlab: True if loading Batlab data for spike detection - DEPRECIATED
	:type initBatlab: boolean
	"""
	def __init__(self, initDirectoryPath, initBatlab=True):
		self.dirPath = initDirectoryPath  # Working directory path
		self.loadBatlab = initBatlab  # DEPRECIATED True if loading Batlab data for spike detection

	def __del__(self):   # Usage: del <instance>
		class_name = self.__class__.__name__
		print class_name, "destroyed"
      
	def ReadMouseSheet(self, excelFileName ):
		""" Read file containing metadata on experimental tests to be loaded.        
		:param excelFileName: Name of Excel file
		:type excelFileName: str
		:returns: pandas.DataFrame with meta data
		"""
		self.mice = pd.read_excel(self.dirPath + '/' + excelFileName, sheetname=0)
		self.mice = self.mice.drop(['oscil', 'pairing', 'notes'], axis=1)  # remove unused columns
		self.mice['mouse_letter'] = self.mice['mouse_letter'].fillna('')
		return self.mice

	def GetUnits(self):
		""" Returns the unique cell identifiers.        
		:param excelFileName: Name of Excel file
		:type excelFileName: str
		:returns: array of strings with unique cell identifiers
		"""
		units = self.mice['mouse_num'].values.astype(str) + self.mice.fillna('')['mouse_letter'] + '_' + self.mice['depth'].values.astype(str)		
		return units

	def GetToneSpikeTrains(self, dataFile, testNumber, spk_threshold=0.3, showTraces=False): # Load a single vocal test without returning stimulation info
		""" Loads recording of microelectrode signals for a tone presentation protocol,
			detects the spikes from threshold crossings, and organizes the spikes in a DataFrame.        
		:param dataFile: path to data file
		:type dataFile: str
		:param testNumber: Test number in the data file with microelectrode signal
		:type testNumber: int
		:param spk_threshold: Threshold for spike detection
		:type spk_threshold: float
		:param showTraces: True if showing all microelectrode signal signal traces (slow)
		:type showTraces: boolean
		:returns: pandas.DataFrame with spike times, duration of recording window
		"""
		test = 'test_'+str(testNumber)
		data = op.open_acqdata(dataFile, filemode='r')
		smplRate = data.get_info(test)['samplerate_ad']
		spikeTrains = pd.DataFrame([])
		nspk = 0
# 		print "WARNING! Maximum number of spikes may limited by first column in DataFrame"
		for t in range(len(data.get_data(test)[0])):
			trace = data.get_data(test)[0][t]
			spikeTimes = 1000*np.array(spst.spike_times(trace, spk_threshold, smplRate))
# 			spikeTrains[str(t)] = pd.Series(spikeTimes)
			spikeTimesS = pd.Series(spikeTimes)
			if spikeTimesS.size > nspk:
				spikeTrains = spikeTrains.reindex(spikeTimesS.index)
				nspk = spikeTimesS.size
			spikeTrains[str(t)] = spikeTimesS
		temptrace = data.get_data(test)[0][0]
		dur = len(temptrace)/float(smplRate)
		if showTraces:
			time = np.linspace(0, dur, num=len(temptrace))
			for t in range(len(data.get_data(test)[0])):
				trace = data.get_data(test)[0][t]
				plt.plot(time, trace)    
		return spikeTrains, dur 

	def GetToneAutoTestData(self, dataFile, testNumber, spk_threshold=0.3):  # Frequency Tuning Curve method
		""" Loads recording of microelectrode signals for multiple tone presentations protocol,
			detects the spikes from threshold crossings, and organizes the spikes in a DataFrame.        
		:param dataFile: path to data file
		:type dataFile: str
		:param testNumber: Test number in the data file with microelectrode signal
		:type testNumber: int
		:param spk_threshold: Threshold for spike detection
		:type spk_threshold: float
		:returns: Dict of pandas.DataFrames with spike times, duration of recording window
		"""
		test = 'test_'+str(testNumber)
		data = op.open_acqdata(dataFile, filemode='r')
		smplRate = data.get_info(test)['samplerate_ad']
		temptrace = data.get_data(test)[0][0]
		duration = len(temptrace)/float(smplRate)
		autoRasters = {}
		numStims = len(data.get_info(test)['stim'])
		for tStim in range(1,numStims):
			freq = int(data.get_info(test)['stim'][tStim]['components'][1]['frequency'])
			attn = int(data.get_info(test)['stim'][tStim]['components'][1]['intensity'])
			traceKey = str(freq)+'_'+str(attn)
			numReps = len(data.get_data(test)[tStim])
			spikeTrains = pd.DataFrame([])
			nspk = 0
# 			print "WARNING! Maximum number of spikes may set by first column in DataFrame"
			for tRep in range(numReps):
				trace = data.get_data(test)[tStim][tRep]
				spikeTimes = 1000*np.array(spst.spike_times(trace, spk_threshold, smplRate))
# 				spikeTrains[str(tRep)] = pd.Series(spikeTimes)
				spikeTimesS = pd.Series(spikeTimes)
				if spikeTimesS.size > nspk:
					spikeTrains = spikeTrains.reindex(spikeTimesS.index)
					nspk = spikeTimesS.size
				spikeTrains[str(tRep)] = spikeTimesS
			autoRasters[traceKey] = spikeTrains
		return autoRasters

	def GetBBNAutoTestData(self, dataFile, testNumber, spk_threshold=0.3):  # Frequency Tuning Curve method
		""" Loads recording of microelectrode signals for multiple broadband noise presentations protocol,
			detects the spikes from threshold crossings, and organizes the spikes in a DataFrame.        
		:param dataFile: path to data file
		:type dataFile: str
		:param testNumber: Test number in the data file with microelectrode signal
		:type testNumber: int
		:param spk_threshold: Threshold for spike detection
		:type spk_threshold: float
		:returns: Dict of pandas.DataFrames with spike times, duration of recording window
		"""
		test = 'test_'+str(testNumber)
		data = op.open_acqdata(dataFile, filemode='r')
		smplRate = data.get_info(test)['samplerate_ad']
		temptrace = data.get_data(test)[0][0]
		duration = len(temptrace)/float(smplRate)
		autoRasters = {}
		numStims = len(data.get_info(test)['stim'])
		for tStim in range(1,numStims):
			attn = int(data.get_info(test)['stim'][tStim]['components'][1]['intensity'])
			traceKey = 'None_'+str(attn)
			numReps = len(data.get_data(test)[tStim])
			spikeTrains = pd.DataFrame([])
			nspk = 0
			for tRep in range(numReps):
				trace = data.get_data(test)[tStim][tRep]
				spikeTimes = 1000*np.array(spst.spike_times(trace, spk_threshold, smplRate))
				spikeTimesS = pd.Series(spikeTimes)
				if spikeTimesS.size > nspk:
					spikeTrains = spikeTrains.reindex(spikeTimesS.index)
					nspk = spikeTimesS.size
				spikeTrains[str(tRep)] = spikeTimesS
			autoRasters[traceKey] = spikeTrains
		return autoRasters

	def GetVocalSpikeTrains(self, dataFile, testNumber, spk_threshold=0.3, showTraces=False): # Load a single vocal test without returning stimulation info
		""" Loads recording of microelectrode signals for a vocalization presentation protocol,
			detects the spikes from threshold crossings, and organizes the spikes in a DataFrame.        
		:param dataFile: path to data file
		:type dataFile: str
		:param testNumber: Test number in the data file with microelectrode signal
		:type testNumber: int
		:param spk_threshold: Threshold for spike detection
		:type spk_threshold: float
		:param showTraces: True if showing all microelectrode signal signal traces (slow)
		:type showTraces: boolean
		:returns: pandas.DataFrame with spike times, duration of recording window
		"""
		test = 'test_'+str(testNumber)
		data = op.open_acqdata(dataFile, filemode='r')
		smplRate = data.get_info(test)['samplerate_ad']
		vocName = data.get_info(test)['stim'][0]['components'][1]['filename']
		spikeTrains = pd.DataFrame([])
		nspk = 0
# 		print "WARNING! Maximum number of spikes may set by first column in DataFrame"
		for t in range(len(data.get_data(test)[0])):
			trace = data.get_data(test)[0][t]
			spikeTimes = 1000*np.array(spst.spike_times(trace, spk_threshold, smplRate))
# 			spikeTrains[str(t)] = pd.Series(spikeTimes)
			spikeTimesS = pd.Series(spikeTimes)
			if spikeTimesS.size > nspk:
				spikeTrains = spikeTrains.reindex(spikeTimesS.index)
				nspk = spikeTimesS.size
			spikeTrains[str(t)] = spikeTimesS
		temptrace = data.get_data(test)[0][0]
		dur = len(temptrace)/float(smplRate)
		if showTraces:
			time = np.linspace(0, dur, num=len(temptrace))
			for t in range(len(data.get_data(test)[0])):
				trace = data.get_data(test)[0][t]
				plt.plot(time, trace)    
		return spikeTrains, vocName, dur 


	def GetVocalResponses(self, dataFile, firstVocal, lastVocal, spk_threshold=0.3):
		""" Wrapper for GetVocalSpikeTrains() to load a range of vocalization tests.        
		:param dataFile: path to data file
		:type dataFile: str
		:param firstVocal: Test number of first vocalization test in the range
		:type firstVocal: int
		:param lastVocal: Test number of last vocalization test in the range
		:type lastVocal: int
		:param spk_threshold: Threshold for spike detection
		:type spk_threshold: float
		:returns: pandas.DataFrame with spike times, duration of recording window
		"""
		vocalSpikesDict = {}
		for vTest in range(int(firstVocal), int(lastVocal)+1):
			spikeTrains, vocalName, dur = self.GetVocalSpikeTrains(dataFile, vTest, spk_threshold=spk_threshold)
			vocalSpikesDict[str(vocalName)] = spikeTrains.stack().reset_index()[0]  # Save only spike timing column for histogram.
		vocalSpikesHisto = pd.DataFrame(vocalSpikesDict)
		vocalNames = vocalSpikesHisto.keys() 
		return vocalSpikesHisto, vocalNames, dur

	def GetFreqsAttns(self, freqTuningHisto):  # Frequency Tuning Curve method
		""" Helper method for ShowSTH() to organize the frequencies in ascending order separated for each intensity.        
		:param freqTuningHisto: dict of pandas.DataFrames with spike data
		:type freqTuningHisto: str
		:returns: ordered list of frequencies (DataFrame keys())
				  numpy array of frequencies
				  numpy array of intensities
		"""
		freqs = np.array([])    
		attns = np.array([])    
		for histoKey in list(freqTuningHisto):
			if histoKey!= 'None_None': 
				freq = histoKey.split('_')[0]
				freqs = np.hstack([freqs, float(freq)/1000])
				attn = histoKey.split('_')[1]
				attns = np.hstack([attns, float(attn)])
		attnCount = stats.itemfreq(attns)
		freqs = np.unique(freqs)
		attns = np.unique(attns)
		if np.max(attnCount[:,1]) !=  np.min(attnCount[:,1]):
			abortedAttnIdx = np.where(attnCount[:,1]!=np.max(attnCount[:,1]))
			attns = np.delete(attns, abortedAttnIdx)
		orderedKeys = []
		for attn in attns:
			freqList = []
			for freq in freqs:
				key = str(int(freq*1000)) + '_' + str(int(attn))
				freqList.append(key)
			orderedKeys.append(freqList)		
		return orderedKeys, freqs, attns

	def GetAttns(self, bbnHisto):  # Broadband noise response method
		""" Helper method for ShowBBNTH() to organize the intensities in ascending order.        
		:param bbnHisto: dict of pandas.DataFrames with spike data
		:type bbnHisto: str
		:returns: ordered list of intensities (DataFrame keys())
				  numpy array of intensities
		"""
		attns = np.array([])    
		for histoKey in list(bbnHisto):
			if histoKey!= 'None_None': 
				attn = histoKey.split('_')[1]
				attns = np.hstack([attns, float(attn)])
		attns = np.unique(attns)        
		orderedKeys = []
		for attn in attns:
			key = 'None_'+str(int(attn))
			orderedKeys.append(key)
		return orderedKeys, attns

	def Raster2Histo(self, freqTuningRaster, duration=250):  # Frequency Tuning Curve method
		""" Helper method for ShowBBNTH() and ShowBBNTH() to make a histogram out of spike times.        
		:param freqTuningRaster: pandas.DataFrames with spike data
		:type freqTuningRaster: pandas.DataFrame
		:param duration: duration of recording window
		:type duration: float
		:returns: pandas.DataFrames of histograms
		"""
		histoDict = {}
		for traceKey in freqTuningRaster.keys():
			histo = np.histogram(freqTuningRaster[traceKey].stack(), bins=int(duration/2), range=(0,duration)) 
			histoDict[traceKey] = histo[0]
		return pd.DataFrame(histoDict)

	def Attn2SPL(self, freq, attn, speakerCalibFileName, sheetName ):
		""" Convert attenuation to sound pressure level for Batlab files using speaker calibration data.        
		:param freq: Frequency of signal
		:type freq: float
		:param attn: Attenuation of signal
		:type attn: float
		:param speakerCalibFileName: Name of file with speaker calibration data
		:type speakerCalibFileName: string
		:param sheetName: Threshold for spike detection
		:type sheetName: float
		:returns: float for sound pressure level
		"""
		speakerCalib = pd.read_excel(self.dirPath + '/' + speakerCalibFileName, sheetName)
		responseCurve = interp1d(speakerCalib['Freq(kHz)'], speakerCalib['SPL(dB)'])
		return responseCurve(freq) - attn

	def PlotRaster(self, spikeTrains, title='', duration=250, figPath=[]):
		""" Plot a raster of spike times versus presentation of the stimulus.        
		:param spikeTrains: pandas.DataFrames with spike data
		:type spikeTrains: pandas.DataFrame
		:param title: Title of plot
		:type title: str
		:param duration: duration of recording window
		:type duration: float
		:param duration: Path to directory where figure is to be saved
		:type duration: str
		:returns: Handle for the figure axis
		"""
		if len(spikeTrains.shape)>1:
			spks = np.array([])
			trns = np.array([])
			for trnNum in range(len(spikeTrains.columns)):
				spkTrn = np.array(spikeTrains.iloc[:,trnNum].dropna())
				trns = np.hstack([trns, (trnNum+1)*np.ones(len(spkTrn))])
				spks = np.hstack([spks, spkTrn])
			#--- Raster plot of spikes ---
			plt.figure(figsize=(8,2))
			sns.set_style("white")
			sns.despine()
			plt.grid(False)
			ax = plt.scatter(spks, trns, marker='s', s=5, color='k')
			plt.ylim(len(spikeTrains.columns)+0.5, 0.5)
			plt.xlim(0, duration)
			plt.xlabel('Time (ms)')
			plt.ylabel('Presentation cycle')
			plt.title(title)
			plt.tick_params(axis='both', which='major', labelsize=14)  
			if len(figPath)>0: plt.savefig(self.dirPath + figPath + 'raster_' + unit + '.png')
		else: print 'Only spike timing information provided, requires presentation numbers for raster.'
		return ax

	def PlotHisto(self, spikeTrains, title='', duration=250, figPath=[]):
		""" Plot a histogram of spike times of the stimulus.        
		:param spikeTrains: pandas.DataFrames with spike data
		:type spikeTrains: pandas.DataFrame
		:param title: Title of plot
		:type title: str
		:param duration: duration of recording window
		:type duration: float
		:param duration: Path to directory where figure is to be saved
		:type duration: str
		:returns: Handle for the figure axis
		"""
		if len(spikeTrains.shape)>1:
			spks = np.array([])
			trns = np.array([])
			for trnNum in range(len(spikeTrains.columns)):
				spkTrn = np.array(spikeTrains.iloc[:,trnNum].dropna())
				trns = np.hstack([trns, (trnNum+1)*np.ones(len(spkTrn))])
				spks = np.hstack([spks, spkTrn])
			spikeTimes = spikeTrains.stack()
		else: spikeTimes = spikeTrains.dropna()
		#--- Histogram of spike times (2 ms bins)---
		sns.set_style("white")
		sns.set_style("ticks")
		plt.figure(figsize=(8,3))
		axHist = spikeTimes.hist(bins=int(duration/2), range=(0,duration))#, figsize=(8,3))
		sns.despine()
		plt.xlim(0, duration)
		plt.xlabel('Time (ms)', size=14)
		plt.ylabel('Number of spikes', size=14)
		plt.title(title)
		plt.tick_params(axis='both', which='major', labelsize=14) 
		plt.grid(False)
		if len(figPath)>0: plt.savefig(self.dirPath + figPath + 'histo_' + title + '.png')
		return axHist

	def ShowSTH(self, freqTuningRaster, unit, duration=250, figPath=[]):  # Frequency Tuning Curve method
		""" Plot a spectral-temporal histogram of spike times versus presentation of the stimulus.        
		:param freqTuningRaster: pandas.DataFrames with spike data
		:type freqTuningRaster: pandas.DataFrame
		:param title: Title of plot
		:type title: str
		:param duration: duration of recording window
		:type duration: float
		:param duration: Path to directory where figure is to be saved
		:type duration: str
		:returns: Handle for the figure axis
		"""
		freqTuningHisto = self.Raster2Histo(freqTuningRaster, duration)
		orderedKeys, freqs, attns = self.GetFreqsAttns(freqTuningHisto)
		fig, ax = plt.subplots(1,len(attns), figsize=(5*len(attns),3), sharey=True)
		if len(attns)>1:
			for p in range(len(attns)):
				ax[p].imshow(freqTuningHisto[orderedKeys[p]].T, extent=[0,duration,min(freqs),max(freqs)], cmap = cm.Greys, aspect='auto', interpolation='nearest', origin='lower')
				ax[p].set_title('Attn '+str(int(attns[p]))+'dB', fontsize=14)
				ax[p].set_xlabel('Time (ms)', size=14)
			ax[0].set_ylabel('Frequency (kHz)', size=14)
			plt.tick_params(axis='both', which='major', labelsize=14)
			plt.tight_layout()    
			fig.text(0.0, 0.92, unit, fontsize=14)
			if len(figPath)>0: plt.savefig(self.dirPath + figPath + 'STH_' + unit + '.png')
		else:
			ax.imshow(freqTuningHisto[orderedKeys[0]].T, extent=[0,duration,min(freqs),max(freqs)], cmap = cm.Greys, aspect='auto', interpolation='nearest', origin='lower')
			ax.set_title('Attn '+str(int(attns))+'dB', fontsize=14)
			ax.set_xlabel('Time (ms)', size=14)
			ax.set_ylabel('Frequency (kHz)', size=14)
			plt.tick_params(axis='both', which='major', labelsize=14)
			plt.tight_layout()    
			fig.text(0.0, 0.92, unit, fontsize=14)
			if len(figPath)>0: plt.savefig(self.dirPath + figPath + 'STH_' + unit + '.png')
		return ax
        
	def ShowBBNTH(self, bbnRaster, unit, duration=250, figPath=[]):  # Intensity versus time histogram for BBN
		""" Plot a SPL-temporal histogram of spike times versus presentation of the stimulus.        
		:param freqTuningRaster: pandas.DataFrames with spike data
		:type freqTuningRaster: pandas.DataFrame
		:param title: Title of plot
		:type title: str
		:param duration: duration of recording window
		:type duration: float
		:param duration: Path to directory where figure is to be saved
		:type duration: str
		:returns: Handle for the figure axis
		"""
		bbnHisto = self.Raster2Histo(bbnRaster, duration)
		fig, ax = plt.subplots(1,1, figsize=(5,3))
		orderedKeys, attns = self.GetAttns(bbnHisto)
		ax.imshow(bbnHisto[orderedKeys].T, extent=[0,duration,min(attns),max(attns)], cmap = cm.Greys, aspect='auto', interpolation='nearest', origin='lower')
		ax.set_title(unit + 'BBN')
		ax.set_xlabel('Time (ms)')
		ax.set_ylabel('Intensity (dB)')
		plt.tight_layout()   
		plt.grid(False) 
		fig.text(0.0, 0.92, unit, fontsize=18)
		if len(figPath)>0: plt.savefig(self.dirPath + figPath + 'ATH_' + unit + '.png')
		return ax
