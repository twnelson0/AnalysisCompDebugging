import awkward as ak
import uproot
import hist
from hist import intervals
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from coffea import processor, nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import candidate, vector
from coffea import util
from math import pi
import numba 
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import vector
import os
import time
import datetime
from distributed import Client
from dask_jobqueue import HTCondorCluster
import csv
#import glob


#X509 function (for HTC)
def move_X509():
	try:
		_x509_localpath = (
			[
				line
				for line in os.popen("voms-proxy-info").read().split("\n")
				if line.startswith("path")
			][0]
			.split(":")[-1]
			.strip()
		)
	except Exception as err:
		raise RuntimeError(
			"x509 proxy could not be parsed, try creating it with 'voms-proxy-init'"
		) from err
	_x509_path = f'/scratch/{os.environ["USER"]}/{_x509_localpath.split("/")[-1]}'
	os.system(f"cp {_x509_localpath} {_x509_path}")
	return os.path.basename(_x509_localpath)


hep.style.use(hep.style.CMS)
TABLEAU_COLORS = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

#Global Variables
WScaleFactor = 1.21
DYScaleFactor = 1.23
TT_FullLep_BR = 0.1061
TT_SemiLep_BR = 0.4392
TT_Had_BR = 0.4544

#Functions and variables for Luminosity weights
lumi_table_data = {"MC Sample":[], "Luminosity":[], "Cross Section (pb)":[], "Number of Events":[], "Calculated Weight":[]}

#Dictionary of cross sections 
xSection_Dictionary = {"Signal": 0.01, #Chosen to make plots readable
						#TTBar Background
						#"TTTo2L2Nu": 831.76*TT_FullLep_BR, "TTToSemiLeptonic": 831.76*TT_SemiLep_BR, "TTToHadronic": 831.76*TT_Had_BR,
						"TTTo2L2Nu": 87.5595, "TTToSemiLeptonic": 365.2482, "TTToHadronic": 381.0923,
						
						#DiBoson Background
						#"ZZ2l2q": 3.22, "WZ3l1nu": 4.708, "WZ2l2q": 5.595, "WZ1l1nu2q": 10.71, "VV2l2nu": 11.95, "WZ1l3nu": 3.05, #"WZ3l1nu.root" : 27.57,
						"ZZ2l2q": 3.676, "WZ2l2q": 6.565, "WZ1l1nu2q": 9.119, "WZ1l3nu": 3.414, "VV2l2nu": 11.09, "WWTo1L1Nu2Q": 51.65, "WWTo4Q": 51.03, "ZZTo4Q": 3.262, "ZZTo2L2Nu": 0.9738, "ZZTo2Nu2Q": 4.545, #"WZ3l1nu.root" : 27.57,
						
						#ZZ->4l
						"ZZ4l": 1.325,
						#DiBoson continued
						#"ZZTo2L2Nu_powheg": 0.564, "ZZTo2L2Q_amcNLO": 3.22, "ZZTo4L_powheg": 1.212, "WWTo2L2Nu_powheg": 12.178, "WWTo4Q_powheg": 51.723, "WWTo1LNuQQ_powheg": 49.997, 
						#"WZTo1L3Nu_amcatnloFXFX": 3.033, "WZTo2L2Q_amcNLO": 5.595, "WZTo3LNu_amcNLO": 4.42965, "WZTo1L1Nu2Q_amcNLO": 10.71, "WW1l1nu2q": 49.997, "WZ1l3nu": 3.05,
						#Single Top Background
						#"Tbar-tchan": 26.23, "T-tchan": 44.07, "Tbar-tW": 35.6, "T-tW": 35.6, 
						"Tbar-tchan": 80.8, "T-tchan": 134.2, "Tbar-tW": 39.65, "T-tW": 39.65, "ST_s-channel_4f_leptonDecays": 3.588, "ST_s-channel_4f_hadronicDecays": 7.485,
						#Drell-Yan Jets
						"DYJetsToLL_M-4to50_HT-70to100": 314.8,
						"DYJetsToLL_M-4to50_HT-100to200": 190.6,
						"DYJetsToLL_M-4to50_HT-200to400": 42.27,
						"DYJetsToLL_M-4to50_HT-400to600": 4.05,
						"DYJetsToLL_M-4to50_HT-600toInf": 1.216,
						"DYJetsToLL_M-50_HT-70to100": 140.0,
						"DYJetsToLL_M-50_HT-100to200": 139.2,
						"DYJetsToLL_M-50_HT-200to400": 38.4,
						"DYJetsToLL_M-50_HT-400to600": 5.174,
						"DYJetsToLL_M-50_HT-600to800": 1.258,
						"DYJetsToLL_M-50_HT-800to1200": 0.5598,
						"DYJetsToLL_M-50_HT-1200to2500": 0.1305,
						"DYJetsToLL_M-50_HT-2500toInf": 0.002997,
						
						#WJets
						"WJetsToLNu_HT-70To100":1283.0, "WJetsToLNu_HT-100To200" : 1244.0, "WJetsToLNu_HT-200To400": 337.8, "WJetsToLNu_HT-400To600": 44.93, "WJetsToLNu_HT-600To800": 11.19, "WJetsToLNu_HT-800To1200": 4.926, "WJetsToLNu_HT-1200To2500" : 1.152, "WJetsToLNu_HT-2500ToInf" : 0.02646, 
						#SM Higgs
						"ZH125": 0.7544*0.0621, "ggZHLL125":0.1223 * 0.062 * 3 * 0.033658, "ggZHNuNu125": 0.1223*0.062*0.2,"ggZHQQ125": 0.1223*0.062*0.6991, "toptopH125": 0.5033*0.062, #"ggH125": 48.30* 0.0621, "qqH125": 3.770 * 0.0621, "WPlusH125": 
						#QCD
						"QCD_HT300to500": 347700, "QCD_HT500to700": 32100, "QCD_HT700to1000": 6831, "QCD_HT1000to1500": 1207, "QCD_HT1500to2000": 119.9, "QCD_HT2000toInf": 25.24,
						}	
Lumi_2018 = 59830

#Dictionary of number of events (values specified in main loop)
numEvents_Dict = {}
sumWEvents_Dict = {}
working_dir = os.getcwd()

def weight_calc(sample,numEvents=1):
	return Lumi_2018*xSection_Dictionary[sample]/numEvents

class PlottingScriptProcessor(processor.ProcessorABC):
	def __init__(self): #Additional arguements can be added later
		self.isData = False #Default assumption is MC
		#pass

	def process(self, events):
		vector.register_awkward()
		#Begin by checking if running on data or sample
		dataset = events.metadata['dataset']
		if ("Data_" in dataset): #Check to see if running on data
			print("Is Data")
			self.isData = True	
		
		event_level = ak.zip(
			{
				"MET_trigger": events.HLT_MET120_IsoTrk50,
				"MET_trigger1": events.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight,
				"MET_trigger2": events.HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight,
				"pfMET": events.MET_pt,
				"pfMETPhi": events.MET_phi,
				"event_weight": ak.ones_like(events.MET_pt), #*0.9,
				"n_electrons": ak.zeros_like(events.MET_pt),
				"n_muons": ak.zeros_like(events.MET_pt),
				"n_tau_electrons": ak.zeros_like(events.MET_pt),
				"n_tau_muons": ak.zeros_like(events.MET_pt),
				"n_tau_hadronic": ak.zeros_like(events.MET_pt),
				"event_num": events.event,
				"run": events.run,
				"Lumi" : events.luminosityBlock,
				#"genWeight": events.genWeight
			},
			with_name="EventArray",
			behavior=candidate.behavior,
		)
		boostedtau = ak.zip( 
			{
				"pt": events.boostedTau_pt,
				"Px": events.boostedTau_pt*np.cos(events.boostedTau_phi),
				"Py": events.boostedTau_pt*np.sin(events.boostedTau_phi),
				"Pz": (events.boostedTau_pt/np.sin(2*np.arctan(np.exp(-events.boostedTau_eta))))*np.cos(2*np.arctan(np.exp(-events.boostedTau_eta))),
				"E": np.sqrt((events.boostedTau_pt/np.sin(2*np.arctan(np.exp(-events.boostedTau_eta))))**2 + events.boostedTau_mass**2),
				"mass": events.boostedTau_mass,
				"eta": events.boostedTau_eta,
				"phi": events.boostedTau_phi,
				"nBoostedTau": events.nboostedTau,
				"charge": events.boostedTau_charge,
				#"iso": events.boostedTau_rawMVAoldDM2017v2,
				"iso": events.boostedTau_idDeepTau2018v2p7VSjet,
				"DBT": events.boostedTau_rawDeepTau2018v2p7VSjet,
				#"decay": events.boostedTaupfTausDiscriminationByDecayModeFinding,
				"decay": events.boostedTau_idDecayModeOldDMs,
			},
			with_name="BoostedTauArray",
			behavior=candidate.behavior,
		)
		tau = ak.zip( 
			{
				"pt": events.Tau_pt,
				"Px": events.Tau_pt*np.cos(events.Tau_phi),
				"Py": events.Tau_pt*np.sin(events.Tau_phi),
				"Pz": (events.Tau_pt/np.sin(2*np.arctan(np.exp(-events.Tau_eta))))*np.cos(2*np.arctan(np.exp(-events.Tau_eta))),
				"E": np.sqrt((events.Tau_pt/np.sin(2*np.arctan(np.exp(-events.Tau_eta))))**2 + events.Tau_mass**2),
				"mass": events.Tau_mass,
				"eta": events.Tau_eta,
				"phi": events.Tau_phi,
			},
			with_name="TauArray",
			behavior=candidate.behavior,
		)
		electron = ak.zip(
			{
				"pt": events.Electron_pt,
				"eta": events.Electron_eta,
				"phi": events.Electron_phi,
				"charge": events.Electron_charge,
				"nElectron": events.nElectron,
				"Px": events.Electron_pt*np.cos(events.Electron_phi),
				"Py": events.Electron_pt*np.sin(events.Electron_phi),
				"Pz": events.Electron_pt*np.tan(2*np.arctan(np.exp(-events.Electron_eta)))**-1,
				"E": np.sqrt(events.Electron_pt**2 + (events.Electron_pt/np.tan(2*np.arctan(np.exp(-events.Electron_eta))))**2 + events.Electron_mass**2),
				#"SCEta": events.Electron_SCEta,
				"SCEta": events.Electron_deltaEtaSC,
				#"IDMVANoIso": events.Electron_IDMVANoIso,
				"IDMVANoIso": events.Electron_mvaNoIso,
					
			},
			with_name="ElectronArray",
			behavior=candidate.behavior,
			
		)
		muon = ak.zip(
			{
				"pt": events.Muon_pt,
				"eta": events.Muon_eta,
				"phi": events.Muon_phi,
				"charge": events.Muon_charge,
				"nMuon": events.nMuon,
				"Px": events.Muon_pt*np.cos(events.Muon_phi),
				"Py": events.Muon_pt*np.sin(events.Muon_phi),
				"Pz": events.Muon_pt*np.tan(2*np.arctan(np.exp(-events.Muon_eta)))**-1,
				"E": np.sqrt(events.Muon_pt**2 + (events.Muon_pt/np.tan(2*np.arctan(np.exp(-events.Muon_eta))))**2 + events.Muon_mass**2),
				"nMu": events.nMuon,
				#"IDbit": events.muIDbit, #No idea what the nanoAOD analog is for this 
				#"IDbit": events.Muon_IDbit,
				"IDSelec": events.Muon_mediumId,
				"D0": events.Muon_dxy,
				"Dz": events.Muon_dz
					
			},
			with_name="MuonArray",
			behavior=candidate.behavior,
			
		)

		AK8Jet = ak.zip(
			{
				"AK8JetDropMass": events.FatJet_msoftdrop,
				#"AK8JetPt": events.FatJet_pt,
				"pt": events.FatJet_pt,
				"eta": events.FatJet_eta,
				"phi": events.FatJet_phi,
				"nAK8Jet": events.nFatJet 
			},
			with_name="AK8JetArray",
			behavior=candidate.behavior,
		)
		
		Jet = ak.zip(
			{
				"pt": events.Jet_pt,
				#"PFLooseId": events.JetPFLooseId,
				"PFLooseId": events.Jet_jetId, #Not sure that this is correct
				"eta": events.Jet_eta,
				"phi": events.Jet_phi,
				"nJet": events.nJet,
				#"DeepCSVTags_b": events.Jet_DeepCSVTags_b
				"DeepCSVTags_b": events.Jet_btagCSVV2,
			},
			with_name="PFJetArray",
			behavior=candidate.behavior,
		)
		
		#tau = tau[ak.argsort(tau.pt,axis=1)] #Force tau pT Ordering
		print("!!!=====Dataset=====!!!!")	
		print(type(dataset))
		print(dataset)

		if not(self.isData):
			event_level["event_weight"] = events.genWeight #Set the event weight to the gen weight

		#Basic Kinematic histograms Boosted tau
		h_boostedtau_pT_Trigger = hist.Hist.new.Regular(50,0,1000,label = r"Boosted-$\tau$ $p_T$ [GeV]").Double()
		h_boostedtau_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"Boosted-$\tau$ $\eta$").Double()
		h_boostedtau_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"Boosted-$\tau$ $\phi$").Double()
		h_boostedtau_raw_iso_Trigger = hist.Hist.new.Regular(20,-1,1,label=r"Raw MVA Score").Double()
		
		#Basic Kinematic histograms Boosted tau
		h_tau_pT_Trigger = hist.Hist.new.Regular(50,0,1000,label = r"$\tau$ $p_T$ [GeV]").Double()
		h_tau_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"$\tau$ $\eta$").Double()
		h_tau_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"$\tau$ $\phi$").Double()

		#Basic Kinematic histograms leptons (muons and electrons)
		h_electron_pT_Trigger = hist.Hist.new.Regular(50,0,1000,label = r"e $p_T$ [GeV]").Double()
		h_electron_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"e $\eta$").Double()
		h_electron_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"e $\phi$").Double()
		h_muon_pT_Trigger = hist.Hist.new.Regular(50,0,1000,label = r"$\mu$ $p_T$ [GeV]").Double()
		h_muon_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"$\mu$ $\eta$").Double()
		h_muon_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"$\mu$ $\phi$").Double()

		#Basic Kinematic histograms Jets (check which Jets most useful based on 
		h_Jet_pT_Trigger = hist.Hist.new.Regular(50,0,1000,label = r"Jet $p_T$ [GeV]").Double() 
		h_Jet_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"Jet $\eta$").Double() 
		h_Jet_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"Jet $\phi$").Double() 
		h_AK8Jet_pT_Trigger = hist.Hist.new.Regular(50,0,1000,label = r"AK8Jet $p_T$ [GeV]").Double() 
		h_AK8Jet_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"AK8Jet $\eta$").Double() 
		h_AK8Jet_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"AK8Jet $\phi$").Double() 

		#Basic Kinematic Histogram for MET
		h_MET_Trigger = hist.Hist.new.Regular(20,0,1000, label = "MET [GeV]").Double()
		
		#Obtain the cross section scale factor	
		if (self.isData):
			CrossSec_Weight = 1 
		else:
			CrossSec_Weight = weight_calc(dataset,sumWEvents_Dict[dataset])
		
		#Fill histograms prior to trigger and all selections (excluding skimming) 
		
		print("Number of events before selection + Trigger: %d"%ak.num(tau,axis=0))
		
		#Apply the trigger
		boostedtau_1 = boostedtau[event_level.MET_trigger1]
		tau_1 = tau[event_level.MET_trigger1]
		AK8Jet_1 = AK8Jet[event_level.MET_trigger1]
		Jet_1 = Jet[event_level.MET_trigger1]
		electron_1 = electron[event_level.MET_trigger1]
		muon_1 = muon[event_level.MET_trigger1]
		event_level_1 = event_level[event_level.MET_trigger1]
		
		boostedtau_noPass = boostedtau[np.bitwise_not(event_level.MET_trigger1)]
		tau_noPass = tau[np.bitwise_not(event_level.MET_trigger1)]
		AK8Jet_noPass = AK8Jet[np.bitwise_not(event_level.MET_trigger1)]
		Jet_noPass = Jet[np.bitwise_not(event_level.MET_trigger1)]
		electron_noPass = electron[np.bitwise_not(event_level.MET_trigger1)]
		muon_noPass = muon[np.bitwise_not(event_level.MET_trigger1)]
		event_level_noPass = event_level[np.bitwise_not(event_level.MET_trigger1)]
		
		boostedtau_2 = boostedtau_noPass[event_level_noPass.MET_trigger2]
		tau_2 = tau_noPass[event_level_noPass.MET_trigger2]
		AK8Jet_2 = AK8Jet_noPass[event_level_noPass.MET_trigger2]
		Jet_2 = Jet_noPass[event_level_noPass.MET_trigger2]
		electron_2 = electron_noPass[event_level_noPass.MET_trigger2]
		muon_2 = muon_noPass[event_level_noPass.MET_trigger2]
		event_level_2 = event_level_noPass[event_level_noPass.MET_trigger2]
		
		#Recombine data
		boostedtau = ak.concatenate((boostedtau_1,boostedtau_2))
		tau = ak.concatenate((tau_1,tau_2))
		AK8Jet = ak.concatenate((AK8Jet_1,AK8Jet_2))
		Jet = ak.concatenate((Jet_1,Jet_2))
		electron = ak.concatenate((electron_1,electron_2))
		muon = ak.concatenate((muon_1,muon_2))
		event_level = ak.concatenate((event_level_1,event_level_2))
		
		#Offline selection
		tau = tau[event_level.pfMET > 180]
		boostedtau = boostedtau[event_level.pfMET > 180]
		AK8Jet = AK8Jet[event_level.pfMET > 180]
		Jet = Jet[event_level.pfMET > 180]
		electron = electron[event_level.pfMET > 180]
		muon = muon[event_level.pfMET > 180]
		event_level = event_level[event_level.pfMET > 180]

		#Fill histograms after to trigger and all selections
		#Boosted Taus
		h_boostedtau_pT_Trigger.fill(ak.ravel(boostedtau.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(boostedtau.pt))[0]))
		h_boostedtau_eta_Trigger.fill(ak.ravel(boostedtau.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(boostedtau.eta))[0]))
		h_boostedtau_phi_Trigger.fill(ak.ravel(boostedtau.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(boostedtau.phi))[0]))
		h_boostedtau_raw_iso_Trigger.fill(ak.ravel(boostedtau.iso),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(boostedtau.iso))[0]))
		
		#Taus
		h_tau_pT_Trigger.fill(ak.ravel(tau.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(tau.pt))[0]))
		h_tau_eta_Trigger.fill(ak.ravel(tau.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(tau.eta))[0]))
		h_tau_phi_Trigger.fill(ak.ravel(tau.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(tau.phi))[0]))

		#Electrons
		h_electron_pT_Trigger.fill(ak.ravel(electron.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(electron.pt))[0]))
		h_electron_eta_Trigger.fill(ak.ravel(electron.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(electron.eta))[0]))
		h_electron_phi_Trigger.fill(ak.ravel(electron.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(electron.phi))[0]))

		#Muons
		h_muon_pT_Trigger.fill(ak.ravel(muon.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(muon.pt))[0]))
		h_muon_eta_Trigger.fill(ak.ravel(muon.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(muon.eta))[0]))
		h_muon_phi_Trigger.fill(ak.ravel(muon.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(muon.phi))[0]))

		#Jets 
		h_Jet_pT_Trigger.fill(ak.ravel(Jet.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(Jet.pt))[0]))
		h_Jet_eta_Trigger.fill(ak.ravel(Jet.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(Jet.eta))[0]))
		h_Jet_phi_Trigger.fill(ak.ravel(Jet.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(Jet.phi))[0]))

		#AK8/Fat Jets
		h_AK8Jet_pT_Trigger.fill(ak.ravel(AK8Jet.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet.pt))[0]))
		h_AK8Jet_eta_Trigger.fill(ak.ravel(AK8Jet.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet.eta))[0]))
		h_AK8Jet_phi_Trigger.fill(ak.ravel(AK8Jet.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet.phi))[0]))

		#Fill MET histogram
		h_MET_Trigger.fill(ak.ravel(event_level.pfMET),weight=ak.ravel(event_level.event_weight*CrossSec_Weight))

		
		return{
			dataset: {
				#"Weight": CrossSec_Weight,
				"Weight_Val": CrossSec_Weight,
				"Weight": ak.to_list(event_level.event_weight*CrossSec_Weight), 
				#Boosted Tau kineamtic distirubtions
				"boostedtau_pt_Trigg": h_boostedtau_pT_Trigger,
				"boostedtau_eta_Trigg": h_boostedtau_eta_Trigger,
				"boostedtau_phi_Trigg": h_boostedtau_phi_Trigger,
				"boostedtau_iso_Trigg": h_boostedtau_raw_iso_Trigger,
				#Tau kineamtic distirubtions
				"tau_pt_Trigg": h_tau_pT_Trigger,
				"tau_eta_Trigg": h_tau_eta_Trigger,
				"tau_phi_Trigg": h_tau_phi_Trigger,
				#Electron kineamtic distirubtions
				"electron_pt_Trigg": h_electron_pT_Trigger,
				"electron_eta_Trigg": h_electron_eta_Trigger,
				"electron_phi_Trigg": h_electron_phi_Trigger,
				#Muon kineamtic distirubtions
				"muon_pt_Trigg": h_muon_pT_Trigger,
				"muon_eta_Trigg": h_muon_eta_Trigger,
				"muon_phi_Trigg": h_muon_phi_Trigger,
				#Jet kineamtic distirubtions
				"Jet_pt_Trigg": h_Jet_pT_Trigger,
				"Jet_eta_Trigg": h_Jet_eta_Trigger,
				"Jet_phi_Trigg": h_Jet_phi_Trigger,
				#AK8Jet kineamtic distirubtions
				"AK8Jet_pt_Trigg": h_AK8Jet_pT_Trigger,
				"AK8Jet_eta_Trigg": h_AK8Jet_eta_Trigger,
				"AK8Jet_phi_Trigg": h_AK8Jet_phi_Trigger,
				#Number of objects of interest
				"Num_tau": ak.sum(ak.num(tau.pt,axis=1)),
				"Num_electron": ak.sum(ak.num(electron.pt,axis=1)),
				"Num_muon": ak.sum(ak.num(muon.pt,axis=1)),
				"Num_Jet": ak.sum(ak.num(Jet.pt,axis=1)),
				"Num_AK8Jet": ak.sum(ak.num(AK8Jet.pt,axis=1)),
				#MET
				"MET": h_MET_Trigger,
			}
		}

	def postprocess(self, accumulator):
		pass

if __name__ == "__main__":
	print("Test Stuff")
	
	#Condor related stuff
	os.environ["CONDOR_CONFIG"] = "/etc/condor/condor_config"
	#Xrootd crap
	_x509_path = move_X509()
	print(f"x509 path: {_x509_path}")
	#htc_log_err_dir = "/scratch/twnelson/ControlPlot_HTC/Run_" + str(time.localtime()[0]) + "_" + str(time.localtime()[1]) + "_" + str(time.localtime()[2]) + "_" + str(time.localtime()[3]) + f".{time.localtime()[4]:02d}"
	#os.makedirs(htc_log_err_dir)

    #DO NOT DELETE THESE LINES CONDOR BROKEN???
	#cluster = ""
#	cluster = HTCondorCluster(
#            cores=1,
#			 memory="5 GB",
#            disk="1.5 GB",
#            death_timeout = '60',
#            job_extra_directives={
#                "+JobFlavour": '"tomorrow"',
#                "log": "dask_job_output.$(PROCESS).$(CLUSTER).log",
#                "output": "dask_job_output.$(PROCESS).$(CLUSTER).out",
#                "error": "dask_job_output.$(PROCESS).$(CLUSTER).err",
#                "should_transfer_files": "yes",
#                "when_to_transfer_ouput": "ON_EXIT_OR_EVICT",
#                "transfer_executable": "false",
#                "+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest-py3.10"',
#                #"+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.25-py3.10"',
#                "Requirements": "HasSingularityJobStart",
#                "InitialDir": f'/scratch/{os.environ["USER"]}',
#                'transfer_input_files': f"{_x509_path}",
#
#            },
#            job_script_prologue = [
#                "export XRD_RUNFORKHANDLER=1",
#                f"export X509_USER_PROXY={_x509_path}",
#            ]
#    )
#	cluster.adapt(minimum=1, maximum=500)

	run_on_condor = False
	
	if (run_on_condor):
		print("Run on Condor")
		iterative_runner = processor.Runner(
			#executor = processor.DaskExecutor(client=Client(cluster)),
			executor = processor.DaskExecutor(client=Client(cluster),status=False),
			schema=BaseSchema,
			skipbadfiles=True,
			xrootdtimeout=1000,
			#executor = processor.DaskExecutor(client=cowtools.GetCondorClient(container_image="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.25-py3.10")),
			#executor = processor.DaskExecutor(client=cowtools.GetCondorClient(cluster,container_image="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.25-py3.10")),
		#executor = processor.IterativeExecutor(compression=None), #This needs to be changed
		)
	else:
		#iterative_runner = processor.Runner(executor = processor.FuturesExecutor(), schema=BaseSchema)
		iterative_runner = processor.Runner(executor = processor.IterativeExecutor(), schema=BaseSchema)
	
	four_tau_hist_list = [
			"boostedtau_pt_Trigg","boostedtau_eta_Trigg","boostedtau_phi_Trigg",
			"tau_pt_Trigg","tau_eta_Trigg","tau_phi_Trigg",
			"electron_pt_Trigg","electron_eta_Trigg","electron_phi_Trigg",
			"muon_pt_Trigg","muon_eta_Trigg","muon_phi_Trigg",
			"Jet_pt_Trigg","Jet_eta_Trigg","Jet_phi_Trigg",
			"AK8Jet_pt_Trigg","AK8Jet_eta_Trigg","AK8Jet_phi_Trigg","MET",
	] 

	#Diretory for files
	Skimmed_Ganesh_base = "root://cmsxrootd.hep.wisc.edu//store/user/gparida/HHbbtt/Hadded_Skimmed_Files/Full_Production_CMSSW_13_0_13_Nov24_23/LooseSelection_MET_gt_80_nFatJet_gt_0_Skim/2018/"
	
	file_dict_test = {
			"ZZ4l": [Skimmed_Ganesh_base + "ZZTo4L.root"],
			"Data_MET": [Skimmed_Ganesh_base + "MET/MET_Run2018A.root"]
        }
	file_dict_data_test = {
			"Data_MET": [Skimmed_Ganesh_base + "MET/MET_Run2018A.root"]
        }
	file_dict_full = {
			"TTToSemiLeptonic": list(np.append([Skimmed_Ganesh_base + "TTToSemiLeptonic_" + str(j) + ".root" for j in range(2,5)], Skimmed_Ganesh_base + "TTToSemiLeptonic.root")),
			"TTTo2L2Nu": [Skimmed_Ganesh_base + "TTTo2L2Nu.root", Skimmed_Ganesh_base + "TTTo2L2Nu_2.root"],
			"TTToHadronic": [Skimmed_Ganesh_base + "TTToHadronic.root"],
			"ZZ4l": [Skimmed_Ganesh_base + "ZZTo4L.root"],
			"ZZTo2L2Nu": [Skimmed_Ganesh_base + "ZZTo2L2Nu.root"],
			"ZZTo2Nu2Q": [Skimmed_Ganesh_base + "ZZTo2Nu2Q_5f.root"],
			"ZZTo4Q": [Skimmed_Ganesh_base + "ZZTo4Q_5f.root"],
			"VV2l2nu": [Skimmed_Ganesh_base + "WWTo2L2Nu.root"],
			"WWTo1L1Nu2Q": [Skimmed_Ganesh_base + "WWTo1L1Nu2Q_4f.root"],
			"WWTo4Q": [Skimmed_Ganesh_base + "WWTo4Q_4f.root"],
			"WZ1l3nu": [Skimmed_Ganesh_base + "WZTo1L3Nu_4f.root"],
			"ZZ2l2q": [Skimmed_Ganesh_base + "ZZTo2Q2L_mllmin4p0.root"],
			"WZ2l2q": [Skimmed_Ganesh_base + "WZTo2Q2L_mllmin4p0.root"],
			"WZ1l1nu2q" : [Skimmed_Ganesh_base + "WZTo1L1Nu2Q_4f.root"],
			"DYJetsToLL_M-4to50_HT-70to100": [Skimmed_Ganesh_base + "DYJetsToLL_M-4to50_HT-70to100.root"],
			"DYJetsToLL_M-4to50_HT-100to200": [Skimmed_Ganesh_base + "DYJetsToLL_M-4to50_HT-100to200.root"],
			"DYJetsToLL_M-4to50_HT-200to400": [Skimmed_Ganesh_base + "DYJetsToLL_M-4to50_HT-200to400.root"],
			"DYJetsToLL_M-4to50_HT-400to600": [Skimmed_Ganesh_base + "DYJetsToLL_M-4to50_HT-400to600.root"],
			"DYJetsToLL_M-4to50_HT-600toInf": [Skimmed_Ganesh_base + "DYJetsToLL_M-4to50_HT-600toInf.root"],
			"DYJetsToLL_M-50_HT-70to100": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-70to100.root"],
			"DYJetsToLL_M-50_HT-100to200": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-100to200.root"],
			"DYJetsToLL_M-50_HT-200to400": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-200to400.root"],
			"DYJetsToLL_M-50_HT-400to600": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-400to600.root"],
			"DYJetsToLL_M-50_HT-600to800": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-600to800.root"],
			"DYJetsToLL_M-50_HT-800to1200": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-800to1200.root"],
			"DYJetsToLL_M-50_HT-1200to2500": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-1200to2500.root"],
			"DYJetsToLL_M-50_HT-2500toInf": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-2500toInf.root"],
			"T-tchan": [Skimmed_Ganesh_base + "ST_t-channel_top_4f_InclusiveDecays.root"],
			"Tbar-tchan": [Skimmed_Ganesh_base + "ST_t-channel_antitop_4f_InclusiveDecays.root"],
			"T-tW": [Skimmed_Ganesh_base + "ST_tW_top_5f_inclusiveDecays.root"],
			"Tbar-tW": [Skimmed_Ganesh_base + "ST_tW_antitop_5f_inclusiveDecays.root"],
			"ST_s-channel_4f_hadronicDecays": [Skimmed_Ganesh_base + "ST_s-channel_4f_hadronicDecays.root"],
			"ST_s-channel_4f_leptonDecays": [Skimmed_Ganesh_base + "ST_s-channel_4f_leptonDecays.root"],
			"WJetsToLNu_HT-70To100": [Skimmed_Ganesh_base + "WJetsToLNu_HT-70To100.root"],
			"WJetsToLNu_HT-100To200": [Skimmed_Ganesh_base + "WJetsToLNu_HT-100To200.root"],
			"WJetsToLNu_HT-200To400": [Skimmed_Ganesh_base + "WJetsToLNu_HT-200To400.root"],
			"WJetsToLNu_HT-400To600": [Skimmed_Ganesh_base + "WJetsToLNu_HT-400To600.root",
				Skimmed_Ganesh_base +"WJetsToLNu_HT-400To600_2.root"],
			"WJetsToLNu_HT-600To800": [Skimmed_Ganesh_base + "WJetsToLNu_HT-600To800.root",
				Skimmed_Ganesh_base + "WJetsToLNu_HT-600To800_2.root"],
			"WJetsToLNu_HT-800To1200": [Skimmed_Ganesh_base + "WJetsToLNu_HT-800To1200.root",
				Skimmed_Ganesh_base + "WJetsToLNu_HT-800To1200_2.root"],
			"WJetsToLNu_HT-1200To2500": [Skimmed_Ganesh_base + "WJetsToLNu_HT-1200To2500.root",
				Skimmed_Ganesh_base + "WJetsToLNu_HT-1200To2500_2.root"],
			"WJetsToLNu_HT-2500ToInf": [Skimmed_Ganesh_base + "WJetsToLNu_HT-2500ToInf.root",
				Skimmed_Ganesh_base + "WJetsToLNu_HT-2500ToInf_2.root"],
			"Data_MET": [Skimmed_Ganesh_base + "MET/MET_Run2018A.root",Skimmed_Ganesh_base + "MET/MET_Run2018B.root",Skimmed_Ganesh_base + "MET/MET_Run2018C.root",
                Skimmed_Ganesh_base + "MET/MET_Run2018D.root",Skimmed_Ganesh_base + "MET/MET_Run2018D_2.root",Skimmed_Ganesh_base + "MET/MET_Run2018D_3.root",
                Skimmed_Ganesh_base + "MET/MET_Run2018D_4.root"]
        }

	file_dict = file_dict_full

	for key_name, file_array in file_dict.items(): 
		print(key_name)
		if (key_name != "Data_MET" ): 
			numEvents_Dict[key_name] = 0 #Initialize the number of events dictionary
			sumWEvents_Dict[key_name] = 0 #Initialize the number of events dictionary
			for file in file_array:
				with uproot.open(file) as tempFile:
					print(file)
					numEvents_Dict[key_name] += np.sum(tempFile['Runs/genEventCount'].array()) #Fixed for nanoAOD 
					sumWEvents_Dict[key_name] += np.sum(tempFile['Runs/genEventSumw'].array()) #Fixed for nanoAOD 
					print(key_name + "sum: %f"%numEvents_Dict[key_name])

		else: #Ignore data files
			numEvents_Dict[key_name] = 1
			sumWEvents_Dict[key_name] = 1

	#Full background list
	background_list_full = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"]
	background_list_test = [r"$ZZ \rightarrow 4l$"]
	background_list_none = []
	background_list = background_list_full
	background_plot_names = {r"$t\bar{t}$" : "_ttbar_", r"$t\bar{t}$ Hadronic" : "_ttbarHadronic_", r"$t\bar{t}$ Semileptonic" : "_ttbarSemilepton_",
			r"$t\bar{t}$ 2L2Nu" : "_ttbar2L2Nu_", r"Drell-Yan+Jets": "_DYJets_", "Di-Bosons" : "_DiBosons_", "Single Top": "_SingleTop_", "QCD" : "_QCD_", 
			"W+Jets" : "_WJets_", r"$ZZ \rightarrow 4l$" : "_ZZ4l_", r"$ZZ \rightarrow 4l$ Test": "_ZZ4lTest_", r"$ZZ \rightarrow 4l$ Control": "_ZZ4lControl_",
			"W+Jets HT 70-100 GeV" : "_WJetsHT70-100_","W+Jets HT 100-200 GeV" : "_WJetsHT100-200_","W+Jets HT 200-400 GeV" : "_WJetsHT200-400_",
			"W+Jets HT 400-600 GeV" : "_WJetsHT400-600_","W+Jets HT 600-800 GeV" : "_WJetsHT600-800_","W+Jets HT 800-1200 GeV" : "_WJetsHT800-1200_",
			"W+Jets HT 1200-2500 GeV" : "_WJetsHT1200-2500_","W+Jets HT 2500-Inf GeV" : "_WJetsHT2500-Inf_"} #For file names
	
	background_dict = {r"$t\bar{t}$" : ["TTToSemiLeptonic","TTTo2L2Nu","TTToHadronic"], r"$t\bar{t}$ Hadronic" : ["TTToHadronic"], 
			r"$t\bar{t}$ Semileptonic" : ["TTToSemiLeptonic"], r"$t\bar{t}$ 2L2Nu" : ["TTTo2L2Nu"],
			r"Drell-Yan+Jets": ["DYJetsToLL_M-4to50_HT-70to100","DYJetsToLL_M-4to50_HT-100to200","DYJetsToLL_M-4to50_HT-200to400","DYJetsToLL_M-4to50_HT-400to600",
			"DYJetsToLL_M-4to50_HT-600toInf","DYJetsToLL_M-50_HT-70to100","DYJetsToLL_M-50_HT-100to200","DYJetsToLL_M-50_HT-200to400",
			"DYJetsToLL_M-50_HT-400to600","DYJetsToLL_M-50_HT-600to800","DYJetsToLL_M-50_HT-800to1200","DYJetsToLL_M-50_HT-1200to2500","DYJetsToLL_M-50_HT-2500toInf"], 
			#"Di-Bosons": ["WZ3l1nu","WZ2l2q","WZ1l1nu2q","ZZ2l2q", "WZ1l3nu", "VV2l2nu"], "Single Top": ["Tbar-tchan","T-tchan","Tbar-tW","T-tW"], 
			"Di-Bosons": ["WZ2l2q","WZ1l1nu2q","ZZ2l2q", "WZ1l3nu", "VV2l2nu", "WWTo1L1Nu2Q", "WWTo4Q", "ZZTo4Q", "ZZTo2L2Nu", "ZZTo2Nu2Q"], 
			"Single Top": ["Tbar-tchan","T-tchan","Tbar-tW","T-tW","ST_s-channel_4f_leptonDecays", "ST_s-channel_4f_hadronicDecays"], 
			"W+Jets": ["WJetsToLNu_HT-70To100","WJetsToLNu_HT-100To200","WJetsToLNu_HT-200To400","WJetsToLNu_HT-400To600","WJetsToLNu_HT-600To800","WJetsToLNu_HT-800To1200","WJetsToLNu_HT-1200To2500","WJetsToLNu_HT-2500ToInf"],
			"W+Jets HT 100-200 GeV": ["WJetsToLNu_HT-100To200"],"W+Jets HT 200-400 GeV": ["WJetsToLNu_HT-200To400"],"W+Jets HT 400-600 GeV": ["WJetsToLNu_HT-400To600"],
			"W+Jets HT 600-800 GeV": ["WJetsToLNu_HT-600To800"],"W+Jets HT 800-1200 GeV": ["WJetsToLNu_HT-800To1200"],
			"W+Jets HT 1200-2500 GeV": ["WJetsToLNu_HT-1200To2500"], "W+Jets HT 2500-Inf GeV": ["WJetsToLNu_HT-2500ToInf"],
			r"$ZZ \rightarrow 4l$" : ["ZZ4l"],
			#r"$ZZ \rightarrow 4l$ Test": ["ZZ4l_Test"],
			#r"$ZZ \rightarrow 4l$ Control": ["ZZ4l_Control"],
	}

	#Dictinary with file names
	trigger_name = "MET_Trigger"
	four_tau_names = {
		"boostedtau_pt_Trigg": "Tau_pT_Trigger" + "-" + trigger_name,
		"boostedtau_eta_Trigg": "Tau_eta_Trigger" + "-" + trigger_name,
		"boostedtau_phi_Trigg": "Tau_phi_Trigger" + "-" + trigger_name,
		"boostedtau_iso_Trigg": "Tau_iso_Trigger" + "-" + trigger_name,
		"tau_pt_Trigg": "Tau_pT_Trigger" + "-" + trigger_name,
		"tau_eta_Trigg": "Tau_eta_Trigger" + "-" + trigger_name,
		"tau_phi_Trigg": "Tau_phi_Trigger" + "-" + trigger_name,
		"electron_pt_Trigg": "Electron_pT_Trigger" + "-" + trigger_name,
		"electron_eta_Trigg": "Electron_eta_Trigger" + "-" + trigger_name,
		"electron_phi_Trigg": "Electron_phi_Trigger" + "-" + trigger_name,
		"muon_pt_Trigg": "Muon_pT_Trigger" + "-" + trigger_name,
		"muon_eta_Trigg": "Muon_eta_Trigger" + "-" + trigger_name,
		"muon_phi_Trigg": "Muon_phi_Trigger" + "-" + trigger_name,
		"Jet_pt_Trigg": "Jet_pT_Trigger" + "-" + trigger_name,
		"Jet_eta_Trigg": "Jet_eta_Trigger" + "-" + trigger_name,
		"Jet_phi_Trigg": "Jet_phi_Trigger" + "-" + trigger_name,
		"AK8Jet_pt_Trigg": "AK8Jet_pT_Trigger" + "-" + trigger_name,
		"AK8Jet_eta_Trigg": "AK8Jet_eta_Trigger" + "-" + trigger_name,
		"AK8Jet_phi_Trigg": "AK8Jet_phi_Trigger" + "-" + trigger_name,
		"MET": "MET_Trigger" + "-" + trigger_name,
	}

	fourtau_out = iterative_runner(file_dict, treename="Events", processor_instance=PlottingScriptProcessor()) #Modified for NanoAOD (changd treename)

	#Dictionaries of histograms for background, signal and data
	hist_dict_background = dict.fromkeys(four_tau_hist_list)
	hist_dict_signal = dict.fromkeys(four_tau_hist_list)
	hist_dict_data = dict.fromkeys(four_tau_hist_list)

	#print("Ran iterative runner")
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	outfile = os.path.join(os.getcwd(), f"output_2018_run{timestamp}.coffea")
	util.save(fourtau_out, outfile)
	print(f"Saved output to {outfile}")	

	for hist_name in four_tau_hist_list: #Loop over all histograms

		temp_hist_dict = dict.fromkeys(background_list) # create dictionary of histograms for each background type
				
		for background_type in background_list:
			print("Background type %s"%background_type)
			background_array = []
			backgrounds = background_dict[background_type]
						
			#Loop over all backgrounds
			for background in backgrounds:
				print("%s"%background)
				if (True): #Only need to generate single background once
					
					#Plot the cutflow for each background
					if (hist_name == "cutflow_table"):
						#print(fourtau_out[background]["cutflow_table"].axes)
						if (background == backgrounds[0]):
							cutflow_hist = fourtau_out[background]["cutflow_table"]
						else:
							cutflow_hist += fourtau_out[background]["cutflow_table"]
							
						if (background == backgrounds[-1]):
							fig2p5, ax2p5 = plt.subplots()
							cutflow_hist.plot1d(ax=ax2p5)
							plt.title(background_type + " Cutflow Table")
							ax2p5.set_yscale('log')
							plt.savefig("SingleBackground" + background_plot_names[background_type] + "CutFlowTable")
							plt.close()
							
						#Plot the weights for each background
						if (background == backgrounds[0]):
							weight_hist = fourtau_out[background]["weight_Hist"]
						else:
							weight_hist += fourtau_out[background]["weight_Hist"]
						if (background == backgrounds[-1]):
							figweight, axweight = plt.subplots()
							weight_hist.plot1d(ax=axweight)
							plt.title(background_type + " Weight Histogram")
							plt.savefig("SingleBackground" + background_plot_names[background_type] + "Weight")
							plt.close()
							
					if (hist_name == "Radion_Charge_Arr"):
						lumi_table_data["MC Sample"].append(background)
						lumi_table_data["Luminosity"].append(fourtau_out[background]["Lumi_Val"])
						lumi_table_data["Cross Section (pb)"].append(fourtau_out[background]["CrossSec_Val"])
						#lumi_table_data["Number of Events"].append(fourtau_out[background]["NEvent_Val"])
						lumi_table_data["Gen SumW"].append(fourtau_out[background]["SumWEvent_Val"])
						lumi_table_data["Calculated Weight"].append(fourtau_out[background]["Weight_Val"])
							
					if (hist_name != "Electron_tau_dR_Arr" and hist_name != "Muon_tau_dR_Arr"):
						if (background == backgrounds[0]):
							crnt_hist = fourtau_out[background][hist_name]
							print("Background: " + background)
							print("Sum of entries: %f"%fourtau_out[background][hist_name].sum())
						else:
							crnt_hist += fourtau_out[background][hist_name]
							print("Background: " + background)
							print("Sum of entries: %f"%fourtau_out[background][hist_name].sum())
						if (background == backgrounds[-1]):
							fig2, ax2 = plt.subplots()
							temp_hist_dict[background_type] = crnt_hist #Try to fix stacking bug
							crnt_hist.plot1d(ax=ax2)
							#if (hist_name == "FourTau_Mass_Arr"):
							print("Background: " + background_type)
							print("Sum of entries: %f"%crnt_hist.sum())
							#print("Number of Entries: %d"%fourtau_out[background]["num_events"])
							plt.title(background_type)
							plt.savefig("SingleBackground" + background_plot_names[background_type] + four_tau_names[hist_name])
							plt.close()

					else: #lepton-tau delta R 
						fig2, ax2 = plt.subplots()
						fourtau_out[background][hist_name].plot1d(ax=ax2)
						ax2.set_yscale('log')
						plt.title(background_type)
						plt.savefig("SingleBackground" + background_plot_names[background_type] + four_tau_names[hist_name])
						plt.close()
						

		#Combine the backgrounds together
		hist_dict_background[hist_name] = hist.Stack.from_dict(temp_hist_dict) #This could be causing the problems 
		
		#Obtain data distributions
		print("==================Hist %s================"%hist_name)
		hist_dict_data[hist_name] = fourtau_out["Data_MET"][hist_name] #.fill("Data",fourtau_out["Data_SingleMuon"][hist_name]) 
		background_stack = hist_dict_background[hist_name] #hist_dict_background[hist_name].stack("background")
		#signal_stack = hist_dict_signal[hist_name].stack("signal")
		
		data_stack = hist_dict_data[hist_name] #.stack("data")    
		#signal_array = [signal_stack["Signal"]]
		data_array = [data_stack] #["Data"]]
				
	#	for background in background_list:
	#		background_array.append(background_stack[background]) #Is this line fucking up your scaling??
	#		print("Background: " + background)
	#		print("Sum of stacked histogram: %f"%background_stack[background].sum())
					
		#Stack background distributions and plot signal + data distribution
		fig,ax = plt.subplots()
		#hep.histplot(background_array,ax=ax,stack=True,histtype="fill",label=background_list,facecolor=TABLEAU_COLORS[:len(background_list)],edgecolor=TABLEAU_COLORS[:len(background_list)])
		#hep.histplot(signal_array,ax=ax,stack=True,histtype="step",label=signal_list,edgecolor=TABLEAU_COLORS[len(background_list)+1],linewidth=2.95)
		hep.histplot(data_array,ax=ax,stack=False,histtype="errorbar", yerr=True,label=["Data"],marker="o",color = "k") #,facecolor='black',edgecolor='black') #,mec='k')
		hep.cms.text("Preliminary",loc=0,fontsize=13)
		ax.set_title("2018 Data",loc = "right")
		ax.legend(fontsize=10, loc='upper right')
		plt.savefig(four_tau_names[hist_name])
		plt.close()

		print("Number of boosted taus: %d"%fourtau_out["Data_MET"]["Num_tau"])
		print("Number of electrons: %d"%fourtau_out["Data_MET"]["Num_electron"])
		print("Number of muons: %d"%fourtau_out["Data_MET"]["Num_muon"])
		print("Number of Jets: %d"%fourtau_out["Data_MET"]["Num_Jet"])
		print("Number of AK8Jets: %d"%fourtau_out["Data_MET"]["Num_AK8Jet"])

