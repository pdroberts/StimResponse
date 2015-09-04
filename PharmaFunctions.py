from IPython.display import display   # To show DataFrames nicely
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize  # to choose figure size
import seaborn as sns                   # Custom graphing commands
import matplotlib.cm as cm                # Color map for STH (gray), etc
import matplotlib.gridspec as gridspec  # For layout of figures
import cPickle     # For saving and loading data structures from disk
from glob import glob   # for getting data file names
import os   # for file system commands
from StimResponse import *   # Custom data analysis commands
from sparkle.data.open import open_acqdata
import sparkle.tools.spikestats as spst
from sparkle.tools import spikestats
ved = VocalEphysData.VocalEphysData('')


def GetSpikeTrains(dataPath, mouse, testNum, spontStim=1, thresh=0.3, plotTrace=[], savePlot=[]):
	""" Read file containing data on experimental tests to be loaded.        
	:param dataPath: Path to directory containing the data directories
	:type dataPath: str
	:returns: spikeTrains, 
			  duration, 
			  startTime, 
			  data
	"""
	dataMouse = mouse['Mouse ID'] + os.sep
	dataFileName = str(mouse['File'])
	dataFile = glob(dataPath + dataMouse + dataFileName + '*')[0]   # Get the absolute path to the file
	fileExt = dataFile.split('.')[1]    # Get the file extension
	if fileExt == 'pst':
		spikeTrains, duration, data = ved.GetToneSpikeTrains2(dataFile, testNum, spk_threshold=thresh, showNumTraces=plotTrace)
		startTime = data.get_info('test_'+str(testNum))['start'][11:-6]
	else:
		if len(plotTrace)>0: print 'opening data file', dataFile
		data = open_acqdata(dataFile, filemode='r')
		# --- Get the segment/test combo ---
		for k in data.hdf5.keys():
			if  'test_'+str(testNum) in data.hdf5[k].keys():
				seg = k
				seg_test = k+'/'+'test_'+str(testNum)
		fs = data.get_info(seg)['samplerate_ad']
		reps = data.get_info(seg_test)['reps']
		if len(plotTrace)>0: print '# reps=', reps, 'sample rate =', fs

		# Get the start time of the test
		startTime = data.get_info(seg_test)['start']
		if len(plotTrace)>0: print 'Start time =', startTime

		# Get the recording trace
		trace_data = data.get_data(seg_test)
		if len(plotTrace)>0: print trace_data.shape
		if len(trace_data.shape) == 4: 
			trace_data = trace_data.squeeze()
		
		# Compute threshold from average maximum of traces
		maxTrace = []
		for n in range(len(trace_data[0,:,0])):
			maxTrace.append( np.max( np.abs(trace_data[spontStim,n,:]) ) )
		aveMax = np.array(maxTrace).mean()
	#         if max(maxTrace) > 1 * np.std(maxTrace):  # remove an extreme outlyer caused by an electrical glitch
	#             maxTrace.remove(max(maxTrace))
		th = 0.7*aveMax
		thresh = th
		spikeTrains = pd.DataFrame([])
		nspk = 0
		for n in range(len(trace_data[0,:,0])): 
			spikes = spikestats.spike_times(trace_data[spontStim,n,:], threshold=thresh, fs=fs)
			spikeTimes = 1000*np.array( spikes )
			spikeTimesS = pd.Series(spikeTimes)
			if spikeTimesS.size > nspk:
				spikeTrains = spikeTrains.reindex(spikeTimesS.index)
				nspk = spikeTimesS.size
			spikeTrains[str(n)] = spikeTimesS
		if len(plotTrace)>0: f = plt.figure()
		for n in plotTrace:
			sns.set_style("white")
			sns.set_style("ticks")
			x = np.arange(len(trace_data[spontStim,n,:]))/fs  # create x axis time points for each sample
			plt.plot(x, trace_data[spontStim,n,:])
			sns.despine()
			plt.grid(False)
			spikes = spikestats.spike_times(trace_data[spontStim,n,:], threshold=thresh, fs=fs)
			plt.plot(spikes, -np.ones_like(spikes)*thresh, 'ok')
			plt.xlabel('time (s)')
		if len(plotTrace)>0 and len(savePlot)>0: 
			unitName = mouse['Mouse ID'] + '_' + str(int(mouse['Depth']))
			plt.savefig(savePlot + unitName +  '_' + str(testNum) + '.png')
		duration = trace_data.shape[-1]/fs
	return spikeTrains, duration, startTime, data

def GetPSTH(dataPath, mice, unitNum, test, spontStim=1, plotTrace=range(0,10)):
    mouse = mice.ix[unitNum]
    if mouse[test] >= ',': 
        tests = mouse[test].split(',')
        testNum = tests[0]
    else: 
        testNum = mouse[test]
    if len(mouse['Mouse ID']) == 9:
        return GetSpikeTrains(dataPath, mouse, testNum=testNum, spontStim=spontStim, thresh=mouse['threshold'], plotTrace=plotTrace)     
    else: 
        print 'Problem noted in spreadsheet:', mouse['Mouse ID']
        return [], 0, 0, []

def GetThreshold(dataPath, mice, unitNum, testNum, traceNum=1):
    mouse = mice.ix[unitNum]
    if len(mouse['Mouse ID']) == 9:
        dataMouse = mouse['Mouse ID'] + os.sep
        dataFileName = str(mouse['File'])
        dataFile = glob(dataPath + dataMouse + dataFileName + '*')[0]   # Get the absolute path to the file
        fileExt = dataFile.split('.')[1]    # Get the file extension
        if fileExt == 'pst':
            thresh = 0
        else:
            data = open_acqdata(dataFile, filemode='r')
            # --- Get the segment/test combo ---
            for k in data.hdf5.keys():
                if  'test_'+str(testNum) in data.hdf5[k].keys():
                    seg = k
                    seg_test = k+'/'+'test_'+str(testNum)
            # Get the recording trace
            trace_data = data.get_data(seg_test)
            if len(trace_data.shape) == 4: 
                trace_data = trace_data.squeeze()
            # Compute threshold from average maximum of traces
            maxTrace = []
            for n in range(len(trace_data[traceNum,:,0])):
                maxTrace.append( np.max( np.abs(trace_data[traceNum,n,:]) ) )
            aveMax = np.array(maxTrace).mean()
    #         if max(maxTrace) > 1 * np.std(maxTrace):  # remove an extreme outlyer caused by an electrical glitch
    #             maxTrace.remove(max(maxTrace))
            thresh = 0.7*aveMax
    return thresh


def ResponseSpontCount( spikeTrains, stimStart=10, stimDuration=50 ):
    spikeTimes = spikeTrains.stack().values
    responseCount = len(spikeTimes[spikeTimes < stimStart+stimDuration+10])
    spontCount = len(spikeTimes[spikeTimes > 100])
    return responseCount, spontCount
    
def ResponseStats( spikeTrains, stimStart=10, stimDuration=50 ):
    dur = 0.001*stimDuration
    responseSpikeCount = []
    spontSpikeCount = []
    for k in spikeTrains.keys():
        spk = spikeTrains[k]
        responseSpikeCount.append(len(spk[spk < stimStart+stimDuration+10])/dur)
        spontSpikeCount.append(len(spk[spk > 100])/0.1)
        if len(responseSpikeCount) > 0:
            responseStats = [np.mean(responseSpikeCount), np.std(responseSpikeCount)]
        else: responseStats = [0,0]
        if len(spontSpikeCount) > 0:
            spontStats = [np.mean(spontSpikeCount), np.std(spontSpikeCount)]
        else: spontStats = [0,0]
    return responseStats
        
def SpontaneousStats( spikeTrains, dur ):
    spontSpikeCount = []
    for k in spikeTrains.keys():
        spk = spikeTrains[k]
        spontSpikeCount.append(len(spk.dropna())/dur)
        if len(spontSpikeCount) > 0:
            spontStats = [np.mean(spontSpikeCount), np.std(spontSpikeCount)]
        else: spontStats = [0,0]
    return spontStats

def Plot_PSTH_DrugEffects(dataPath, mouse, tests, plotHisto=False, plotRaster=False, plotTrace=range(0), figPath=[]):
    if len(mouse['Mouse ID']) == 9:
        testNums = []
        for t in tests:
            if isinstance(mouse[t], int): # Check if int, then append to list of testNums
                testNums.append( mouse[t] )
            else:                       # If not int, then assume comma-delinated string, convert to integers, insert into testNums list
                testNums.extend( map(int, mouse[t].split(',')) )  
        testNums = filter(lambda x: x != 0, testNums)  # remove all occurances of 0 from the list of test numbers
        testNums.sort()    # put testNums into ascending (temporal) order
        mouseName = mouse['Mouse ID'] + '_' + str(int(mouse['Depth']))
        date = str(mouse['File'])+'-'
        print 'unit:', mouseName, 'tests:', testNums
        timeStamp = []
        spontAve = []
        spontSTD = []
        responseAve = []
        responseSTD = []
        for t in testNums:
            spikeTrains, dur, time, data = GetSpikeTrains(dataPath, mouse, testNum=t, spontStim=0, thresh=mouse['threshold'], plotTrace=range(0))
            spontStats = SpontaneousStats( spikeTrains, dur )
            spikeTrains, dur, time, data = GetSpikeTrains(dataPath, mouse, testNum=t, spontStim=1, thresh=mouse['threshold'], \
                                                          plotTrace=plotTrace, savePlot=figPath)
            #--- Compute spike statsistics ---
            responseStats = ResponseStats( spikeTrains )
    #         print 'test', t, 'time:', time
    #         print 'Response rate =', responseStats[0], '(', format(responseStats[1], '.2f'), ')', ', spontaneous rate =', spontStats[0], '(', spontStats[1], ')'
            timeStamp.append(date+time)
            spontAve.append(spontStats[0])
            spontSTD.append(spontStats[1])
            responseAve.append(responseStats[0])
            responseSTD.append(responseStats[1])
            # --- Plot raster ---
            if plotRaster:
                ved.PlotRaster(spikeTrains, title=str(t)+', '+time, duration=dur*1000)
                if len(figPath)>0: 
                    plt.savefig(dataPath + figPath + 'raster_' + mouseName + '_' +str(t)+ '_' + mouse['Drug Type'] + '.png')
            # --- Plot histogram ---
            if plotRaster:
                ved.PlotHisto(spikeTrains, title=str(t)+', '+time, duration=dur*1000)
                if len(figPath)>0: 
                    plt.savefig(dataPath + figPath + 'histo_' + mouseName + '_' +str(t)+ '_' + mouse['Drug Type'] + '.png')
        # --- Plot the time dependent change in rates --- 
        timeOn, timeOff = GetPharmaTimes(mouse)
        rateEffects = pd.DataFrame({'Spontaneous': spontAve, 'spontSTD': spontSTD, \
                                    'Response': responseAve, 'responseSTD': responseSTD}, \
                                   index=pd.to_datetime(timeStamp))   
        if len(testNums)>1:
            fig = plt.figure()
            rateEffects['Response'].plot(yerr=rateEffects['responseSTD'], capthick=1)
            rateEffects['Spontaneous'].plot(yerr=rateEffects['spontSTD'], capthick=1)
            plt.legend(loc='upper right', fontsize=12, frameon=True)
            sns.despine()
            plt.grid(False)
            plt.xlabel('Time (ms)', size=14)
            plt.ylabel('Rate (Hz)', size=14)
            plt.title(mouseName +  ', ' + mouse['Drug Type'], size=14)
            plt.tick_params(axis='both', which='major', labelsize=14)
            ax = plt.gca()
            lineY = ax.get_ylim()[1] - 0.01*ax.get_ylim()[1]
            plt.plot((timeOn, timeOff), (lineY, lineY), 'k-',linewidth=4)
            if timeOff==timeOn:
                plt.plot((timeOn, timeOff), (lineY, lineY), 'ko',linewidth=4)
            for i, n in enumerate(testNums):
                plt.annotate(str(n), (rateEffects.index[i],rateEffects['Spontaneous'][i]))
            if len(figPath)>0: 
                plt.savefig(figPath + 'effects_' + mouseName +  ', ' + mouse['Drug Type'] + '.png')
        return rateEffects
    else: 
        print 'Problem noted in spreadsheet:', mouse['Mouse ID']
        return []

def GetPharmaTimes(mouse):
    date = str(mouse['File'])+'-'
    timeOn = mouse['Drugs On Time']
    timeOff = mouse['Drugs Off Time']
    if isinstance(timeOn, unicode):
        if  timeOn[-2:]=='am':
            timeOn = pd.to_datetime(date+timeOn[:-2])
        if timeOn[-2:]=='pm':
            pmHour = int(timeOn[:-2].split(':')[0])
            if pmHour==12: pmHour = str(pmHour)
            else: pmHour = str(pmHour + 12)
            timeOn = pd.to_datetime(date + pmHour + timeOn[:-2].split(':')[1])
    if isinstance(timeOff, unicode):
        if  timeOff[-2:]=='am':
            timeOff = pd.to_datetime(date+timeOff[:-2])
        if timeOff[-2:]=='pm':
            pmHour = int(timeOff[:-2].split(':')[0])
            if pmHour==12: pmHour = str(pmHour)
            else: pmHour = str(pmHour + 12)
            timeOff = pd.to_datetime(date + pmHour + timeOff[:-2].split(':')[1])
    if timeOff==0: timeOff = timeOn
    return timeOn, timeOff

#--- Frequency Tuning curve: ---
def GetTuningData(dataPath, mice, test, unitNum):
    rasters = []
    durs = []
    tests = []
    mouse = mice.ix[unitNum]
    thresh=mouse['threshold']
    if mouse[test] >= ',': 
        tests = mouse[test].split(',')
        testNums = tests
    else: 
        testNums = [mouse[test]]
    for testNum in testNums:
        if len(mouse['Mouse ID']) == 9 and testNum != 0:# and mouse['Type'] == 's':
            dataMouse = mouse['Mouse ID'] + os.sep
            dataFileName = str(mouse['File'])
            dataFile = glob(dataPath + dataMouse + dataFileName + '*')[0]   # Get the absolute path to the file
            fileExt = dataFile.split('.')[1]    # Get the file extension
            if fileExt == 'pst':
                autoRasters, duration, data = ved.GetToneAutoTestData2(dataFile, testNum, spk_threshold=thresh)
                startTime = data.get_info('test_'+str(testNum))['start'][11:-6]
            else:
                data = open_acqdata(dataFile, filemode='r')
                # --- Get the segment/test combo ---
                for k in data.hdf5.keys():
                    if  'test_'+str(testNum) in data.hdf5[k].keys():
                        seg = k
                        segTest = seg +'/'+'test_'+ str(testNum)
                fs = data.get_info(seg)['samplerate_ad']
                reps = data.get_info(segTest)['reps']
                #--- Get the start time of the test
                startTime = data.get_info(segTest)['start']
                #--- Get the recording trace
                trace_data = data.get_data(segTest)
                if len(trace_data.shape) == 4: 
                    trace_data = trace_data.squeeze() 
                stimInfo = eval(data.get_info(segTest)['stim'])
                duration = trace_data.shape[-1]/fs*1000
                autoRasters = {}
                for tStim in range(1,len(stimInfo)):
                    freq = int(stimInfo[tStim]['components'][0]['frequency'])
                    spl = int(stimInfo[tStim]['components'][0]['intensity'])
                    traceKey = str(freq)+'_'+str(spl)
                    numReps = trace_data.shape[1]                    
                    spikeTrains = pd.DataFrame([])
                    nspk = 0
                    for tRep in range(numReps):
                        trace = data.get_data(segTest)[tStim][tRep].squeeze()
                        spikeTimes = 1000*np.array(spst.spike_times(trace, thresh, fs))
                        spikeTimesS = pd.Series(spikeTimes)
                        if spikeTimesS.size > nspk:
                            spikeTrains = spikeTrains.reindex(spikeTimesS.index)
                            nspk = spikeTimesS.size
                        spikeTrains[str(tRep)] = spikeTimesS
                    autoRasters[traceKey] = spikeTrains 
            rasters.append(autoRasters)
            durs.append(duration)
        else: 
            print 'Problem noted in spreadsheet:', mouse['Mouse ID']
    return rasters, durs, testNums

def PlotTuningCurves(mice, unitNum, stRaster, testNum=0, figPath=[]):
        mouse = mice.ix[unitNum]
        tuning = []
        orderedKeys, freqs, spls = ved.GetFreqsAttns(stRaster)
        for s in range(len(orderedKeys)):
            for k in orderedKeys[s]:
                freq = int(k.split('_')[0])
                spl = int(k.split('_')[1])
                raster = stRaster[k]
                res = ResponseStats( raster )
                tuning.append({'intensity': spl, 'freq': freq/1000, 'response': res[0], 'responseSTD': res[1]})
        tuningCurves = pd.DataFrame(tuning)
        db = np.unique(tuningCurves['intensity'])
        fig = plt.figure()
        for d in db:
            tuningCurves[tuningCurves['intensity']==d].plot(x='freq', y='response', yerr='responseSTD', capthick=1, label=str(d)+'dB')
        plt.legend(loc='upper right', fontsize=12, frameon=True)
        sns.despine()
        plt.grid(False)
        plt.xlabel('Frequency (kHz)', size=14)
        plt.ylabel('Response Rate (Hz)', size=14)
        mouseName = mouse['Mouse ID'] + '_' + str(int(mouse['Depth']))
        plt.title(mouseName +  ', ' + mouse['Drug Type'], size=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        if len(figPath)>0: 
            plt.savefig(figPath + 'tuningCurve_' + mouseName +  '_' + mouse['Drug Type'] + '_test_'+str(testNum)+'.png')
        return tuningCurves

def TuningResponseArea(tuningCurves, unitKey='', figPath=[]):
    f = plt.figure()
    colorRange = (-10,10.1)
    I = np.unique(np.array(tuningCurves['intensity']))
    F = np.array(tuningCurves['freq'])
    R = np.array(np.zeros((len(I), len(F))))
    for ci, i in enumerate(I):
        for cf, f in enumerate(F):
            R[ci,cf] = tuningCurves['response'].where(tuningCurves['intensity']==i).where(tuningCurves['freq']==f).dropna().values[0]
    levelRange = np.arange(colorRange[0], colorRange[1], (colorRange[1]-colorRange[0])/float(25*(colorRange[1]-colorRange[0]))) 
    sns.set_context(rc={"figure.figsize": (7, 4)})
    ax = plt.contourf(F, I, R)#, vmin=colorRange[0], vmax=colorRange[1], levels=levelRange, cmap = cm.bwr )
    plt.colorbar()
    # plt.title(unit, fontsize=14)
    plt.xlabel('Frequency (kHz)', fontsize=14)
    plt.ylabel('Intensity (dB)', fontsize=14)
    if len(figPath)>0: 
        plt.savefig(figPath + 'tuningArea_' + unitKey +'.png')

#--- Broad band noise responses and threshold ---
def GetIOData(dataPath, mice, tests, unitNum):
    mouse = mice.ix[unitNum]
    duration = 200
    if len(mouse['Mouse ID']) == 9: # and testNum != 0 and mouse['Type'] == 's':
        thresh=mouse['threshold']     # use this threshold for .pst files and as default

        testNums = []
        for t in tests:
            if isinstance(mouse[t], int): # Check if int, then append to list of testNums
                testNums.append( mouse[t] )
            else:                       # If not int, then assume comma-delinated string, convert to integers, insert into testNums list
                testNums.extend( map(int, mouse[t].split(',')) )  
        testNums = filter(lambda x: x != 0, testNums)  # remove all occurances of 0 from the list of test numbers
        testNums.sort()    # put testNums into ascending (temporal) order
        
        dataMouse = mouse['Mouse ID'] + os.sep
        dataFileName = str(mouse['File'])
        dataFile = glob(dataPath + dataMouse + dataFileName + '*')[0]   # Get the absolute path to the file
        fileExt = dataFile.split('.')[1]    # Get the file extension

        mouseName = mouse['Mouse ID'] + '_' + str(int(mouse['Depth']))
        date = str(mouse['File'])+'-'
        print 'unit:', mouseName, 'tests:', testNums
        rasterSet = {}
        for t in testNums:
            if fileExt == 'pst':
                autoRasters, duration, data = ved.GetBBNAutoTestData(dataFile, t, spk_threshold=thresh)
                startTime = data.get_info('test_'+str(t))['start'][11:-6]
                rasterSet[mouseName +'_'+ str(t)] = autoRasters
            else:
                data = open_acqdata(dataFile, filemode='r')
                # --- Get the segment/test combo ---
                for k in data.hdf5.keys():
                    if  'test_'+str(t) in data.hdf5[k].keys():
                        seg = k
                        segTest = seg +'/'+'test_'+ str(t)
                fs = data.get_info(seg)['samplerate_ad']
                reps = data.get_info(segTest)['reps']
                # Get the start time of the test
                startTime = data.get_info(segTest)['start']
                # Get the recording trace
                trace_data = data.get_data(segTest)
                if len(trace_data.shape) == 4: 
                    trace_data = trace_data.squeeze() 
                stimInfo = eval(data.get_info(segTest)['stim'])
                duration = trace_data.shape[-1]/fs*1000
                thresh = GetThreshold(dataPath, mice, unitNum, t, traceNum=-1) # Calculate thresh from last (loudest) stimulation
                autoRasters = {}
                for tStim in range(1,len(stimInfo)):
                    spl = int(stimInfo[tStim]['components'][0]['intensity'])
                    traceKey = 'None_'+str(spl).zfill(2)    # None_ is for ShowBBNTH() in VocalEphysData.py
                    numReps = trace_data.shape[1]
                    spikeTrains = pd.DataFrame([])
                    nspk = 0
                    for tRep in range(numReps):
                        trace = data.get_data(segTest)[tStim][tRep].squeeze()
                        spikeTimes = 1000*np.array(spst.spike_times(trace, thresh, fs))
                        spikeTimesS = pd.Series(spikeTimes)
                        if spikeTimesS.size > nspk:
                            spikeTrains = spikeTrains.reindex(spikeTimesS.index)
                            nspk = spikeTimesS.size
                        spikeTrains[str(tRep)] = spikeTimesS
                    autoRasters[traceKey] = spikeTrains
                rasterSet[str(unitNum)+'_'+mouseName +'_'+ str(t)] = autoRasters
        return rasterSet, duration
    else: 
        print 'Problem noted in spreadsheet:', mouse['Mouse ID']
        return [], 0

def PlotIOCurve(stRaster, rasterKey, figPath=[]):
    tuning = []
    sortedKeys = sorted(stRaster.keys())
    for traceKey in sortedKeys:
        spl = int(traceKey.split('_')[-1])
        raster = stRaster[traceKey]
        res = ResponseStats( raster )
        tuning.append({'intensity': spl, 'response': res[0], 'responseSTD': res[1]})
    tuningCurves = pd.DataFrame(tuning)
    testNum = int(rasterKey.split('_')[-1])
    tuningCurves.plot(x='intensity', y='response', yerr='responseSTD', capthick=1, label='test '+str(testNum))
    plt.legend(loc='upper left', fontsize=12, frameon=True)
    sns.despine()
    plt.grid(False)
    plt.xlabel('Intensity (dB)', size=14)
    plt.ylabel('Response Rate (Hz)', size=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    title = rasterKey.split('_')[0]+'_'+rasterKey.split('_')[1]+'_'+rasterKey.split('_')[2]
    plt.title(title, size=14)
    if len(figPath)>0: 
        plt.savefig(figPath + 'ioCurves_' + title +'.png')
    return tuningCurves

