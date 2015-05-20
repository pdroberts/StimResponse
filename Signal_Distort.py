#09/11/2014, update 05/18/2015
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import lfilter, firwin, resample, decimate
from IPython.core.pylabtools import figsize
import seaborn as sns
sns.set_style("white")
import matplotlib.cm as cm     # Color map
from utilities.wav_utility import pcm2float
 

class  Signal_Distort:   
    """Class for generating intermodulation distortions in sound files
    	Includes tools for filtering and plotting
    
    :param initVocalDir: Path to directory containing sound files
    :type initVocalDir: str
    :param samplerate: Sampling frequency (Hz) at which data points are collected
    :type samplerate: int
    """
    def __init__( self, initVocalDir, sampleRate = 333000):
    	self.vocalDir = initVocalDir
        self.Fs = sampleRate # (/s)
        self.xs = 26     # (nm)
        self.a1 = 0.065  # (1/nm)
        self.a2 = 0.016  # (1/nm)
        self.x1 = 24     # (nm)
        self.x2 = 41     # (nm)
        self.R_b = 50    # (MOhm)
        self.R_c = 500   # (MOhm)
        self.E_b = 90    # (mV)
        self.E_e = 80    # (mV)

    #---- Lukashkin signal distortion methods ------------------------
    def boltz(self, x, x1_set = -0.2):  # Boltzman function from (Frank and Kossell, 1996).
        x1= x1_set #-0.2 #-0.06 
        x2= x1
        a1 = 12.8
        a2 = a1/3
        return 1./( 1+np.exp( a2*(x2-x) ) * ( 1+np.exp( a1*(x1-x) ) ) )    
    
    def Lukashkin( self, stim, xs = 26):
        self.xs = xs     # (nm)
        dur = len(stim)
        y = np.zeros(dur)
        t = np.zeros(dur)
        for simTime in range(dur):
            t[simTime] = simTime/self.Fs
            y[simTime] = self.Vi(stim[simTime])
        return y

    def Vi( self, x):
    #  basolateral membrane voltage, Eq (4).
        v = (self.E_e + self.E_b)/(1+self.R_a(x)/self.R_b) - self.E_b; 
        return v

    def R_a( self, y):
    #  transducer resistance.    
        Ra = self.R_c + (1 + np.exp(self.a2*(self.x2-self.xs-y))*(1+np.exp(self.a1*(self.x1-self.xs-y))));
        return Ra 
    
    def G_tr( self, y):
    #  transducer conductance, Eq1, [Lukashkin98]. 
        G_tr_max = 7  #(nS)
        G_tr = G_tr_max / (1 + np.exp(self.a2*(self.x2-y))*(1+np.exp(self.a1*(self.x1-y))));
        return G_tr 
    
    #---- Signal generation methods ------------------------
    def Stim_2Tone( self, omega_1 = 70000, A_1 = 30, omega_2 = 50000, A_2 = 30, tspan = 0.1 ):
    #  Simulate a tone pair.    
        #Fs = 44100     # sample rate: 44.1 kHz, standard wav file sample rate
        #Fs = 333000     # sample rate: 333 kHz, standard Batlab sample rate
        Fs = self.Fs
        dur = int(tspan*Fs)
        stim = np.zeros(dur)
        for simTime in range(dur):
            stim[simTime] = self.ampl(simTime/Fs, A_1)*np.cos(2*np.pi*omega_1*simTime/Fs)  + self.ampl(simTime/Fs, A_2)*np.cos(2*np.pi*omega_2*simTime/Fs);
        return stim/np.max(stim), Fs 

    def ampl( self, t, A):     #  Helper function for Stim_2Tone.   
        if t < 5: amp = A
        elif (t >= 5) & (t<10): amp = 0
        else: amp = 0
        return amp
    
    #---- Other signal manipulation methods ------------------------
    def ShiftExp_2tone(t, f1, f2, overlap, shift, tau): # overlap is in percent, negative means gap
        A1 = 0.5*np.exp(-shift/tau)
        A2 = 0.5*np.exp(-shift/tau)
        T = len(t)
        sft = shift*(t[1]-t[0])*np.ones(T)
        t = t - sft
        z = np.zeros(T)
        t1 = t[:T/2.0 + 0.01*overlap*T/2.0  + shift ]
        t2 = t[ T/2.0 - 0.01*overlap*T/2.0 + shift:]
        s1 = np.hstack( (A1*np.sin(2*np.pi*f1*t1), z[ T/2.0 + 0.01*overlap*T/2.0 + shift:]) )
        s2 = np.hstack( (z[:T/2.0 - 0.01*overlap*T/2.0 + shift], A2*np.sin(2*np.pi*f2*t2)) )
        return s1 + s2 
    
    def signalReverb(self, sig, shift, tau, echos): # units for shift (!=0) and tau are ms
        sig2 = sig
        for n in range(echos):
            A2 = np.exp(-(n+1)*shift/tau)
            z = np.zeros(int((n+1)*shift*(self.Fs*0.001)))
            sig2 = sig2 + np.hstack( (z, A2*sig[:-np.size(z)]) )
        sig2 = sig2*np.max(sig)/np.max(sig2)
        return sig2 

    def BandPassFilter(self, signal, lowCut=8000.0, highCut=120000.0, set_numtaps=71):
        # Create a FIR filter and apply it to signal.
        nyq_rate = self.Fs / 2.0  # The Nyquist rate of the signal.
        cutoff_hz = np.array([lowCut, highCut]) # The cutoff frequency of the filter:
        numtaps = set_numtaps # Length of the filter (number of coefficients, i.e. the filter order + 1)
        # Use firwin to create a lowpass FIR filter
        fir_coeff = firwin(numtaps, cutoff_hz/nyq_rate, pass_zero=False)
        # Use lfilter to filter the signal with the FIR filter
        return lfilter(fir_coeff, 1.0, signal)
    
    def DistortionProducts(self, signal, reverbShift=1., reverbTau=1., echos=5, distortFactor=4.):
        shift = reverbShift
        tau = reverbTau
        N = echos
        sigReverb = self.signalReverb(signal, shift, tau, N)

        resampleFactor = 4.
        sigReverb = resample(sigReverb, resampleFactor*len(sigReverb))
        self.Fs = resampleFactor*self.Fs

        sigReverbD = self.boltz(sigReverb/distortFactor)
        sigReverbDn = (sigReverbD - np.mean(sigReverbD[:100]))/np.max(np.abs(sigReverbD - np.mean(sigReverbD[:100])))
        sigReverbD_F = self.BandPassFilter(sigReverbDn, set_numtaps=501)

        resampleFactor = 4
        sigReverbD_F = decimate(sigReverbD_F, resampleFactor)
        self.Fs = self.Fs/float(resampleFactor)
        return sigReverbD_F
    
    def PlotSpectrogram(self, signal, figsize=(9,5), title='Spectrogram', specThreshold=-110, zmax=None):
        fig = plt.figure()
        fig.set_size_inches(figsize[0],figsize[1])
        gs = gridspec.GridSpec(2, 2, width_ratios=[1,8], height_ratios=[4,1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax4 = plt.subplot(gs[3]) 
        #--- Spectrogram ---
        nfft = 2**8
        win = np.blackman(nfft) # numpy.blackman(), numpy.hamming(), numpy.bartlett()
        Pxx, freqs, bins, im = ax2.specgram(signal, Fs=self.Fs, NFFT=nfft, window=win, vmin=specThreshold, vmax=zmax, cmap = cm.CMRmap)
        ax2.set_title(title, fontsize=14)
        # ax2.set_ylabel('Frequency (Hz)')
        ax2.set_ylim((0,120000))  
        ax2.set_xlim((0,len(signal)/float(self.Fs)))  
#         ax2.set_xlim((0,0.25))  
        ax2.axis('off')
#         cbar = fig.colorbar(ax2)
        #--- Signal waveform ---
        t = np.linspace(0, len(signal)/float(self.Fs), len(signal))
        ax4.plot(t, signal, 'b')
        # ax4.set_title('Waveform')
        # ax4.set_ylabel('Amplitude')
        ax4.set_xlabel('Time (s)')
        ax4.set_xlim((0,len(signal)/float(self.Fs)))  
#         ax4.set_xlim((0,0.25))  
        ax4.set_ylim([-1, 1])  
        ax4.tick_params(axis='x', top='off')
        ax4.tick_params(axis='y', right='off', labelleft='off')
        #--- Power density plot ---
        ax1.plot(np.log10(Pxx.sum(1)), freqs/1000., 'r')
#         print min(np.log10(Pxx.sum(1))),   max(np.log10(Pxx.sum(1)))  # ===  output size to adjust x-lim  ===
        ax1.set_ylabel('Freq (kHz)')
        # ax1.set_xlabel('Intensity (dB)')
        ax1.set_ylim([0, 120])
        ax1.set_xlim([-12, 0])
        ax1.tick_params(axis='x', top='off', bottom='on', labelbottom='off')
        ax1.tick_params(axis='y', right='off')
        fig.tight_layout(pad=-2.0, w_pad=-5.0, h_pad=-2.0)
        return fig #, Pxx, freqs
        

    def PlotSpectResponse(self, vocalName, miceVocalSpikesDict, unit):
		fileName = self.vocalDir + vocalName
		vocalSignal = np.fromfile(fileName, dtype='int16')
		maxAbs_pcm = max(abs(vocalSignal))  # Capture largest integer in the original .call1 file (int16)
		nVocalSignal = pcm2float(vocalSignal, np.float32)
		sigReverbD_F = nVocalSignal
		t = np.linspace(0, len(sigReverbD_F)/float(self.Fs), len(sigReverbD_F))
		f, axarr = plt.subplots(2, 1, figsize=(10,6))
		# print self.Fs
		Pxx, freqs, bins, im = axarr[0].specgram(sigReverbD_F, NFFT=2**8, Fs=self.Fs, noverlap=10, vmin=-105, cmap = cm.CMRmap)
		axarr[0].set_title(vocalName)
		axarr[0].set_xlabel('Time (s)')
		axarr[0].set_ylabel('Frequency (Hz)')
		axarr[0].set_ylim((0,120000))
		axarr[0].set_xlim((0,len(sigReverbD_F)/float(self.Fs)+0.06))  
# 		v  = vocalName[:-14]+vocalName[-6:]
		v  = vocalName
		axarr[1] = miceVocalSpikesDict[unit][v].hist(bins=125)
		plt.xlabel('Time (ms)')
		plt.ylabel('Number of spikes')
		plt.xlim(0,len(sigReverbD_F)/float(self.Fs)*1000+60)
		plt.show()
    


