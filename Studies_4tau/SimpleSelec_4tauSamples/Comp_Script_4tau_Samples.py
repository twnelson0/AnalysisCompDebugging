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
import glob
import json

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
						"TTTo2L2Nu": 87.5595, "TTToSemiLeptonic": 365.2482, "TTToHadronic": 381.0923,
						
						#DiBoson Background
						"ZZ2l2q": 3.676, "WZ2l2q": 6.565, "WZ1l1nu2q": 9.119, "WZ1l3nu": 3.414, "VV2l2nu": 11.09, "WWTo1L1Nu2Q": 51.65, "WWTo4Q": 51.03, "ZZTo4Q": 3.262, "ZZTo2L2Nu": 0.9738, "ZZTo2Nu2Q": 4.545, #"WZ3l1nu.root" : 27.57,
						
						#ZZ->4l
						"ZZ4l": 1.325,
						
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
						"QCD_HT50to100": 186100000.0, "QCD_HT100to200": 23630000.0, "QCD_HT200to300": 1554000.0, "QCD_HT300to500": 323800.0, "QCD_HT500to700": 30280.0, 
						"QCD_HT700to1000": 6392.0, "QCD_HT1000to1500": 1118.0, "QCD_HT1500to2000": 108.9, "QCD_HT2000toInf": 21.93,   
						}	
Lumi_2018 = 59830

#Dictionary of number of events (values specified in main loop)
numEvents_Dict = {}
sumWEvents_Dict = {}
working_dir = os.getcwd()

def weight_calc(sample,numEvents=1):
	return Lumi_2018*xSection_Dictionary[sample]/numEvents

#def cleaning_func(vec_1, vec_2, limit: float) -> bool:
#	return vec_1.deltaR(vec_2) <= limit

def crossClean(part1, part2, limitVal):
	#Split up particle and cross clean particle
	part1_Vec = ak.zip({"t": part1.E,"x": part1.Px,"y": part1.Py, "z": part1.Pz}, with_name="LorentzVector") 
	part2_Vec = ak.zip({"t": part2.E,"x": part2.Px,"y": part2.Py, "z": part2.Pz}, with_name="LorentzVector") 

	#Crosss Clean 
	cross_clean = ak.all(part1_Vec.metric_table(part2_Vec) <= limitVal,axis=-1)
	
	return cross_clean 

def crossClean_PartJet(part1, Jet, limitVal):
	#Set up 4 vectors
	Jet_x = Jet.pt*np.cos(Jet.phi)
	Jet_y = Jet.pt*np.sin(Jet.phi)
	Jet_z = Jet.pt*np.tan(np.arctan(np.exp(-Jet.eta)))**-1
	Jet_t = np.sqrt(Jet.mass**2 + Jet_x**2 + Jet_y**2 + Jet_z**2)
	
	part1_Vec = ak.zip({"t": part1.E,"x": part1.Px,"y": part1.Py, "z": part1.Pz}, with_name="LorentzVector") 
	Jet_Vec = ak.zip({"t": Jet_t,"x":Jet_x,"y": Jet_y, "z":Jet_z}, with_name="LorentzVector") 

	#Crosss Clean 
	cross_clean = ak.all(part1_Vec.metric_table(Jet_Vec) <= limitVal,axis=-1)
	
	return cross_clean 

def crossClean_JetPart(Jet, Part, limitVal):
	#Set up 4 vectors
	Jet_x = Jet.pt*np.cos(Jet.phi)
	Jet_y = Jet.pt*np.sin(Jet.phi)
	Jet_z = Jet.pt*np.tan(np.arctan(np.exp(-Jet.eta)))**-1
	Jet_t = np.sqrt(Jet.mass**2 + Jet_x**2 + Jet_y**2 + Jet_z**2)

	Jet_Vec = ak.zip({"t": Jet_t,"x": Jet_x,"y": Jet_y, "z": Jet_z}, with_name="LorentzVector") 
	Part_Vec = ak.zip({"t": Part.E,"x": Part.Px,"y": Part.Py, "z": Part.Pz}, with_name="LorentzVector") 

	#Crosss Clean 
	cross_clean = ak.all(Jet_Vec.metric_table(Part_Vec) <= limitVal,axis=-1)
	
	return cross_clean 

def crossClean_DiJet(Jet1, Jet2, limitVal):
	#Set up 4 vectors
	Jet1_x = Jet1.pt*np.cos(Jet1.phi)
	Jet1_y = Jet1.pt*np.sin(Jet1.phi)
	Jet1_z = Jet1.pt*np.tan(np.arctan(np.exp(-Jet1.eta)))**-1
	Jet1_t = np.sqrt(Jet1.mass**2 + Jet1_x**2 + Jet1_y**2 + Jet1_z**2)
	
	Jet2_x = Jet2.pt*np.cos(Jet2.phi)
	Jet2_y = Jet2.pt*np.sin(Jet2.phi)
	Jet2_z = Jet2.pt*np.tan(np.arctan(np.exp(-Jet2.eta)))**-1
	Jet2_t = np.sqrt(Jet2.mass**2 + Jet2_x**2 + Jet2_y**2 + Jet2_z**2)

	Jet1_Vec = ak.zip({"t": Jet1_t,"x": Jet1_x,"y": Jet1_y, "z":Jet1_z}, with_name="LorentzVector") 
	Jet2_Vec = ak.zip({"t": Jet2_t,"x": Jet2_x,"y": Jet2_y, "z":Jet2_z}, with_name="LorentzVector") 

	#Crosss Clean 
	cross_clean = ak.all(Jet1_Vec.metric_table(Jet2_Vec) <= limitVal,axis=-1)
	
	return cross_clean 

def lead_crossClean(particle, crossclean_part, limitVal):
	#Set up the four vectors
	part_4vec = ak.zip({"rho": particle.pt, "phi": particle.phi, "eta": particle.eta ,"tau": particle.mass},with_name="Momentum4D")
	crossclean_4vec = ak.zip({"rho": crossclean_part.pt, "phi": crossclean_part.phi, "eta": crossclean_part.eta, "tau": crossclean_part.mass},with_name="Momentum4D")
	parts,crossclean = ak.unzip(ak.cartesian([part_4vec,crossclean_4vec],axis=1,nested=True))
	deltaR_Arr = parts.deltaR(crossclean)

	#Get the selection and apply it
	selec = deltaR_Arr <= limitVal
	selec = ak.all(selec,axis=2)
	particle = particle[selec]

	return particle

def deltaR_Selec(part1,part2,upp_lim):
	#Obtain 4 vectors
	part1_4vec = ak.zip({"rho": part1.pt, "phi": part1.phi, "eta": part1.eta, "tau": part1.mass},with_name="Momentum4D")
	part2_4vec = ak.zip({"rho": part2.pt, "phi": part2.phi, "eta": part2.eta, "tau": part2.mass},with_name="Momentum4D")
	
	#Obtain seperation
	#deltaR_Arr = part1.deltaR(part2)
	deltaR_Arr = np.sqrt((part1.eta - part2.eta)**2 + ((part1_4vec.phi - part2_4vec.phi + np.pi) %(2*np.pi) - np.pi)**2)

	dR_Cond = deltaR_Arr <= upp_lim

	return dR_Cond

def dimass_Selec(part1,part2,low_lim):
	#Obtain 4 vectors	
	part1_4vec = ak.zip({"rho": part1.pt, "phi": part1.phi, "eta": part1.eta, "tau": part1.mass},with_name="Momentum4D")
	part2_4vec = ak.zip({"rho": part2.pt, "phi": part2.phi, "eta": part2.eta, "tau": part2.mass},with_name="Momentum4D")

	#Combine vectors into di-particle object
	dipart_vector = part1_4vec + part2_4vec

	mass_Cond = dipart_vector.mass > low_lim

	return mass_Cond 

def deltaPhi_METSelec(part1,MET,low_lim): #Note definition of deltaR coppied from coffea code (line 67 of vector.py)
	return (part1.phi - MET.MET_Phi + np.pi) % (2*np.pi) - np.pi > low_lim


class PlottingScriptProcessor(processor.ProcessorABC):
	def __init__(self, nBoostedTaus = 0, ApplyTrigger = True): #Additional arguements can be added later
		self.isData = False #Default assumption is MC
		self.nTau_Selec = nBoostedTaus #Number of tau selections
		self.ApplyTrigger = ApplyTrigger
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
				"METHTMHT_Trigger": events.HLT_PFHT500_PFMET100_PFMHT100_IDTight,
				"Mu_Trigger": events.HLT_Mu50,
				"MET_trigger1": events.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight,
				"MET_trigger2": events.HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight,
				"MET_pt": events.MET_pt,
				"MET_Phi": events.MET_phi,
				"event_weight": ak.ones_like(events.MET_pt), #*0.9,
				"n_electrons": ak.zeros_like(events.MET_pt),
				"n_muons": ak.zeros_like(events.MET_pt),
				"n_tau_electrons": ak.zeros_like(events.MET_pt),
				"n_tau_muons": ak.zeros_like(events.MET_pt),
				"n_tau_hadronic": ak.zeros_like(events.MET_pt),
				"event_num": events.event,
				"run": events.run,
				"Lumi" : events.luminosityBlock,
				"PV_ndof": events.PV_ndof,
				"PV_z": events.PV_z,
				"PV_x": events.PV_x,
				"PV_y": events.PV_y,
				"nFatJet": events.nFatJet,
				"Flag_goodVertices": events.Flag_goodVertices,
				"Flag_globalSuperTightHalo2016Filter": events.Flag_globalSuperTightHalo2016Filter,
				"Flag_HBHENoiseFilter": events.Flag_HBHENoiseFilter,
				"Flag_HBHENoiseIsoFilter": events.Flag_HBHENoiseIsoFilter,
				"Flag_EcalDeadCellTriggerPrimitiveFilter": events.Flag_EcalDeadCellTriggerPrimitiveFilter,
				"Flag_BadPFMuonFilter": events.Flag_BadPFMuonFilter,
				"Flag_BadPFMuonDzFilter": events.Flag_BadPFMuonDzFilter,
				"Flag_hfNoisyHitsFilter": events.Flag_hfNoisyHitsFilter,
				"Flag_eeBadScFilter": events.Flag_eeBadScFilter,
				"Flag_ecalBadCalibFilter": events.Flag_ecalBadCalibFilter,
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
				"nBoostedTau": events.nTau,
				"charge": events.Tau_charge,
				"IDVsJets": events.Tau_idDeepTau2018v2p5VSjet,
				"IDVsEle": events.Tau_idDeepTau2018v2p5VSe,
				"IDVsMu": events.Tau_idDeepTau2018v2p5VSmu,
			},
			with_name="BoostedTauArray",
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
				"mass": events.Electron_mass, 
				#"SCEta": events.Electron_SCEta,
				"SCEta": events.Electron_deltaEtaSC,
				#"IDMVANoIso": events.Electron_IDMVANoIso,
				"IDMVANoIso": events.Electron_mvaNoIso,
				"RelIso": events.Electron_pfRelIso03_all,
				#"Id": events.Electron_looseId,
					
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
				"mass": events.Muon_mass, 
				#"IDbit": events.muIDbit, #No idea what the nanoAOD analog is for this 
				#"IDbit": events.Muon_IDbit,
				"IDSelec_Med": events.Muon_mediumId,
				"D0": events.Muon_dxy,
				"Dz": events.Muon_dz,
				"LooseId": events.Muon_looseId,
				"RelIso": events.Muon_pfRelIso04_all,
				#"RelIso_03": events.Muon_pfRelIso03_all,
					
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
				"nAK8Jet": events.nFatJet,
				"softDropM": events.FatJet_msoftdrop,
				"Id": events.FatJet_jetId,
				"mass": events.FatJet_mass, 
			},
			with_name="AK8JetArray",
			behavior=candidate.behavior,
		)
		
		Jet = ak.zip(
			{
				"pt": events.Jet_pt,
                #"PFLooseId": events.JetPFLooseId,
				"JetId": events.Jet_jetId, #Not sure that this is correct
				"eta": events.Jet_eta,
				"phi": events.Jet_phi,
				"mass": events.Jet_mass,
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
			print("Is MC events are equal to gen weights")
			event_level["event_weight"] = events.genWeight #Set the event weight to the gen weight

		#Basic Kinematic histograms Boosted tau
		h_boostedtau_pT_Trigger = hist.Hist.new.Regular(20,0,400,label = r"Boosted $\tau$ $p_T$ [GeV]").Double()
		h_Leadingboostedtau_pT_Trigger = hist.Hist.new.Regular(20,0,400,label = r"Boosted $\tau$ Leading $p_T$ [GeV]").Double()
		h_Subleadingboostedtau_pT_Trigger = hist.Hist.new.Regular(20,0,400,label = r"Boosted $\tau$ Subleading $p_T$ [GeV]").Double()
		h_Thirdleadingboostedtau_pT_Trigger = hist.Hist.new.Regular(20,0,400,label = r"Boosted $\tau$ 3rd-leading $p_T$ [GeV]").Double()
		h_Fourthleadingboostedtau_pT_Trigger = hist.Hist.new.Regular(20,0,400,label = r"Boosted $\tau$ 4th-leading $p_T$ [GeV]").Double()
		h_boostedtau_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"Boosted $\tau$ $\eta$").Double()
		h_boostedtau_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"Boosted$\tau$ $\phi$").Double()
		h_boostedtau_raw_iso_Trigger = hist.Hist.new.Regular(20,-1,1,label=r"Raw MVA Score").Double()
		
		#Basic Kinematic histograms of tau/HPS tau
		h_tau_pT_Trigger = hist.Hist.new.Regular(20,0,400,label = r" $\tau$ $p_T$ [GeV]").Double()
		h_Leadingtau_pT_Trigger = hist.Hist.new.Regular(20,0,400,label = r"$\tau$ Leading $p_T$ [GeV]").Double()
		h_tau_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r" $\tau$ $\eta$").Double()
		h_tau_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"$\tau$ $\phi$").Double()
		
		#Basic Kinematic histograms leptons (muons and electrons)
		h_electron_pT_Trigger = hist.Hist.new.Regular(15,0,300,label = r"e $p_T$ [GeV]").Double()
		h_Leadingelectron_pT_Trigger = hist.Hist.new.Regular(15,0,300,label = r"e Leading $p_T$ [GeV]").Double()
		h_electron_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"e $\eta$").Double()
		h_electron_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"e Leading $\phi$").Double()
		h_muon_pT_Trigger = hist.Hist.new.Regular(15,0,300,label = r"$\mu$ $p_T$ [GeV]").Double()
		h_Leadingmuon_pT_Trigger = hist.Hist.new.Regular(15,0,300,label = r"$\mu$ Leading $p_T$ [GeV]").Double()
		h_muon_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"$\mu$ $\eta$").Double()
		h_Leadingmuon_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"$\mu$ $\eta$").Double()
		h_muon_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"$\mu$ $\phi$").Double()
		
		#Basic Kinematic histograms Jets (check which Jets most useful based on 
		h_Jet_pT_Trigger = hist.Hist.new.Regular(50,0,700,label = r"Jet $p_T$ [GeV]").Double()
		h_LeadingJet_pT_Trigger = hist.Hist.new.Regular(50,0,700,label = r"Jet Leading $p_T$ [GeV]").Double()
		h_Jet_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"Jet $\eta$").Double()
		h_Jet_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"Jet $\phi$").Double()
		h_AK8Jet_pT_Trigger = hist.Hist.new.Regular(50,0,700,label = r"AK8Jet $p_T$ [GeV]").Double()
		h_LeadingAK8Jet_pT_Trigger = hist.Hist.new.Regular(50,0,700,label = r"AK8Jet Leading $p_T$ [GeV]").Double()
		h_AK8Jet_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"AK8Jet $\eta$").Double()
		h_AK8Jet_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"AK8Jet $\phi$").Double()
		
		#Add MET, HT and MHT histogram
		h_MET_Trigger = hist.Hist.new.Regular(20,0,500,label=r"MET [GeV]").Double()
		h_HT_Trigger = hist.Hist.new.Regular(40,0,1200,label=r"HT [GeV]").Double()
		h_MHT_Trigger = hist.Hist.new.Regular(20,0,500,label=r"MHT [GeV]").Double()

		#Add cutflow and N-1 tables
		if (self.ApplyTrigger):
			h_CutFlow = hist.Hist.new.StrCategory(["SkimOnly","METCut","nFatJetReq","FlagReq","PVSelec","LeadingTau","SubleadingTau","3rdLeadingTau","4thLeadingTau","Trigger"]).Double()
			h_NMinus1 = hist.Hist.new.StrCategory(["SkimOnly","METCut","nFatJetReq","FlagReq","PVSelec","LeadingTau","SubleadingTau","3rdLeadingTau","4thLeadingTau","Trigger"]).Double()
		else:
			h_CutFlow = hist.Hist.new.StrCategory(["SkimOnly","METCut","nFatJetReq","FlagReq","PVSelec","LeadingTau","SubleadingTau","3rdLeadingTau","4thLeadingTau"]).Double()
			h_NMinus1 = hist.Hist.new.StrCategory(["SkimOnly","METCut","nFatJetReq","FlagReq","PVSelec","LeadingTau","SubleadingTau","3rdLeadingTau","4thLeadingTau"]).Double()

		#Fill initial entries in skim and N-1 histograms
		n_Skim = np.size(event_level.nFatJet)
		h_CutFlow.fill("SkimOnly",weight=n_Skim)
		h_NMinus1.fill("SkimOnly",weight=0)
		
		#Obtain the cross section scale factor	
		#CrossSec_Weight = 1 
		if (self.isData):
			CrossSec_Weight = 1 
		else:
			CrossSec_Weight = weight_calc(dataset,sumWEvents_Dict[dataset])

		#Obtain MHT
		Jet_MHT = Jet[Jet.pt > 30]
		Jet_MHT = Jet_MHT[np.abs(Jet_MHT.eta) < 5]
		Jet_MHT = Jet_MHT[Jet_MHT.JetId > 0.5]
		event_level["MHT_x"] = ak.sum(Jet_MHT.pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False) 
		event_level["MHT_y"] = ak.sum(Jet_MHT.pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False)
		event_level["MHT"] = np.sqrt(event_level.MHT_x**2 + event_level.MHT_y**2)
		del Jet_MHT
		
		#Obtain HT
		Jet_HT = Jet[Jet.pt > 30]
		Jet_HT = Jet_HT[np.abs(Jet_HT.eta) < 3]
		Jet_HT = Jet_HT[Jet_HT.JetId > 0.5]
		event_level["HT"] = ak.sum(Jet_HT.pt, axis=1, keepdims=False) 
		del Jet_HT

		#############
        #Cut Selections
        #############

		#MET selection
		tau = tau[event_level.MET_pt > 100]
		boostedtau = boostedtau[event_level.MET_pt > 100]
		AK8Jet = AK8Jet[event_level.MET_pt > 100]
		Jet = Jet[event_level.MET_pt > 100]
		electron = electron[event_level.MET_pt > 100]
		muon = muon[event_level.MET_pt > 100]
		event_level = event_level[event_level.MET_pt > 100]	

		#Fill post MET entries in skim and N-1 histograms
		n_MET = np.size(event_level.nFatJet)
		h_CutFlow.fill("METCut",weight=n_MET)
		h_NMinus1.fill("METCut",weight=n_Skim - n_MET)	

		#Impose all events have at least one fat Jet
		tau = tau[event_level.nFatJet > 0]
		boostedtau = boostedtau[event_level.nFatJet > 0]
		AK8Jet = AK8Jet[event_level.nFatJet > 0]
		Jet = Jet[event_level.nFatJet > 0]
		electron = electron[event_level.nFatJet > 0]
		muon = muon[event_level.nFatJet > 0]
		event_level = event_level[event_level.nFatJet > 0]	

		#Fill post FatJet entries in skim and N-1 histograms
		n_FatJet = np.size(event_level.nFatJet)
		h_CutFlow.fill("nFatJetReq",weight=n_FatJet)
		h_NMinus1.fill("nFatJetReq",weight=n_MET - n_FatJet)

		#Flag conditions
		Flag_Array = ["Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter", "Flag_BadPFMuonDzFilter", "Flag_hfNoisyHitsFilter", "Flag_eeBadScFilter", "Flag_ecalBadCalibFilter"]
		flag_cond = event_level[Flag_Array[0]] #Initialize the condition as the first flag since logical and it with itself will act like an identiy operator
		
		for flag in Flag_Array:
			flag_cond = flag_cond & event_level[flag]
		
		tau = tau[flag_cond]
		boostedtau = boostedtau[flag_cond]
		AK8Jet = AK8Jet[flag_cond]
		Jet = Jet[flag_cond]
		electron = electron[flag_cond]
		muon = muon[flag_cond]
		event_level = event_level[flag_cond]	

		#Fill post flag selections entries in skim and N-1 histograms
		n_FlagSelec = np.size(event_level.nFatJet)
		h_CutFlow.fill("FlagReq",weight=n_FlagSelec)
		h_NMinus1.fill("FlagReq",weight=n_FatJet - n_FlagSelec)

		#PV selections
		ndof_cond = event_level.PV_ndof > 4
		PVz_cond = np.abs(event_level.PV_z) < 24
		PVr_cond = np.sqrt(event_level.PV_x**2 + event_level.PV_y**2) < 2
		PV_Cond = np.bitwise_and(ndof_cond,np.bitwise_and(PVz_cond,PVr_cond))
		
		tau = tau[PV_Cond]
		boostedtau = boostedtau[PV_Cond]
		AK8Jet = AK8Jet[PV_Cond]
		Jet = Jet[PV_Cond]
		electron = electron[PV_Cond]
		muon = muon[PV_Cond]
		event_level = event_level[PV_Cond]	

		#Fill post PV selection entries in skim and N-1 histograms
		n_PVSelec = np.size(event_level.nFatJet)
		h_CutFlow.fill("PVSelec",weight=n_PVSelec)
		h_NMinus1.fill("PVSelec",weight=n_FlagSelec - n_PVSelec)

		n_PreTrigger = n_PVSelec #Set number of events left before trigger seleciton to PV selection	

		#Temp values of the Tau selections
		n_LeadTau = -1
		n_SubLeadTau = -1
		n_3rdLeadTau = -1
		n_4thLeadTau = -1

        #Boosted tau selections
		if (self.nTau_Selec > 0):
			#Impose selections on leading boosted tau
			pT_Cond = boostedtau.pt > 20
			eta_Cond = np.abs(boostedtau.eta) < 2.3
			decayMode_Cond = boostedtau.decay >= 0.5
			DBT_Iso_Cond = boostedtau.DBT >= 0.5
			
			tau_selec_cond = pT_Cond & eta_Cond & decayMode_Cond & DBT_Iso_Cond
			boostedtau[tau_selec_cond] #Apply selections to all individual taus
		
			#Require events have at least 1 boosted tau
			lead_tau_cond = ak.num(boostedtau,axis=1) >= 1
				
			tau = tau[lead_tau_cond]
			boostedtau = boostedtau[lead_tau_cond]
			AK8Jet = AK8Jet[lead_tau_cond]
			Jet = Jet[lead_tau_cond]
			electron = electron[lead_tau_cond]
			muon = muon[lead_tau_cond]
			event_level = event_level[lead_tau_cond]

			#Fill post leading tau selection entries in skim and N-1 histograms
			n_LeadTau = np.size(event_level.nFatJet)
			h_CutFlow.fill("LeadingTau",weight=n_LeadTau)
			h_NMinus1.fill("LeadingTau",weight=n_PVSelec - n_LeadTau)

			n_PreTrigger = n_LeadTau				
			
			#Impose selections on Subleading boosted tau
			if (self.nTau_Selec > 1):
				#Require events have at least 2 boosted tau
				sublead_tau_cond = ak.num(boostedtau,axis=1) >= 2
					
				tau = tau[sublead_tau_cond]
				boostedtau = boostedtau[sublead_tau_cond]
				AK8Jet = AK8Jet[sublead_tau_cond]
				Jet = Jet[sublead_tau_cond]
				electron = electron[sublead_tau_cond]
				muon = muon[sublead_tau_cond]
				event_level = event_level[sublead_tau_cond]

				#Fill post subl-leading tau selection entries in skim and N-1 histograms
				n_SubLeadTau = np.size(event_level.nFatJet)
				h_CutFlow.fill("SubleadingTau",weight=n_SubLeadTau)
				h_NMinus1.fill("SubleadingTau",weight=n_LeadTau - n_SubLeadTau)

				n_PreTrigger = n_SubLeadTau				
			
			#Impose selections on third-leading boosted tau
			if (self.nTau_Selec > 2):
				#Require events have at least 2 boosted tau
				thirdlead_tau_cond = ak.num(boostedtau,axis=1) >= 3
					
				tau = tau[thirdlead_tau_cond]
				boostedtau = boostedtau[thirdlead_tau_cond]
				AK8Jet = AK8Jet[thirdlead_tau_cond]
				Jet = Jet[thirdlead_tau_cond]
				electron = electron[thirdlead_tau_cond]
				muon = muon[thirdlead_tau_cond]
				event_level = event_level[thirdlead_tau_cond]

				#Fill post 3rd leading tau selection entries in skim and N-1 histograms
				n_3rdLeadTau = np.size(event_level.nFatJet)
				h_CutFlow.fill("3rdLeadingTau",weight=n_3rdLeadTau)
				h_NMinus1.fill("3rdLeadingTau",weight=n_SubLeadTau - n_3rdLeadTau)

				n_PreTrigger = n_3rdLeadTau				
			
			#Impose selections on fourth-leading boosted tau
			if (self.nTau_Selec > 3):
				#Require events have at least 2 boosted tau
				fourthlead_tau_cond = ak.num(boostedtau,axis=1) >= 4
					
				tau = tau[fourthlead_tau_cond]
				boostedtau = boostedtau[fourthlead_tau_cond]
				AK8Jet = AK8Jet[fourthlead_tau_cond]
				Jet = Jet[fourthlead_tau_cond]
				electron = electron[fourthlead_tau_cond]
				muon = muon[fourthlead_tau_cond]
				event_level = event_level[fourthlead_tau_cond]

				#Fill post 4th leading tau selection entries in skim and N-1 histograms
				n_4thLeadTau = np.size(event_level.nFatJet)
				h_CutFlow.fill("4thLeadingTau",weight=n_4thLeadTau)
				h_NMinus1.fill("4thLeadingTau",weight=n_3rdLeadTau - n_4thLeadTau)

				n_PreTrigger = n_4thLeadTau				
		
		#############
        #Trigger and Offline Cuts
        #############
		
		if (self.ApplyTrigger):
			#HLT Trigger(s)
			boostedtau = boostedtau[event_level.Mu_Trigger]
			tau = tau[event_level.Mu_Trigger]
			AK8Jet = AK8Jet[event_level.Mu_Trigger]
			Jet = Jet[event_level.Mu_Trigger]
			electron = electron[event_level.Mu_Trigger]
			muon = muon[event_level.Mu_Trigger]
			event_level = event_level[event_level.Mu_Trigger]

			#Muon Trigger offline selection
			tau = tau[ak.any(muon.nMu > 0, axis = 1)]
			boostedtau = boostedtau[ak.any(muon.nMu > 0, axis = 1)]
			AK8Jet = AK8Jet[ak.any(muon.nMu > 0, axis = 1)]
			Jet = Jet[ak.any(muon.nMu > 0, axis = 1)]
			electron = electron[ak.any(muon.nMu > 0, axis = 1)]
			muon = muon[ak.any(muon.nMu > 0, axis = 1)]
			event_level = event_level[ak.any(muon.nMu > 0, axis = 1)]				

			tau = tau[ak.any(muon.pt > 52, axis=1)]
			boostedtau = boostedtau[ak.any(muon.pt > 52, axis=1)]
			AK8Jet = AK8Jet[ak.any(muon.pt > 52, axis=1)]
			Jet = Jet[ak.any(muon.pt > 52, axis=1)]
			electron = electron[ak.any(muon.pt > 52, axis=1)]
			muon = muon[ak.any(muon.pt > 52, axis=1)]
			event_level = event_level[ak.any(muon.pt > 52, axis=1)]	

			#Apply isolation and ID selections on muons
			id_selec = muon[:,0].IDSelec_Med
			Iso_selec = muon[:,0].RelIso < 0.10
			
			tau = tau[np.bitwise_and(id_selec,Iso_selec)]
			boostedtau = boostedtau[np.bitwise_and(id_selec,Iso_selec)]
			AK8Jet = AK8Jet[np.bitwise_and(id_selec,Iso_selec)]
			Jet = Jet[np.bitwise_and(id_selec,Iso_selec)]
			electron = electron[np.bitwise_and(id_selec,Iso_selec)]
			muon = muon[np.bitwise_and(id_selec,Iso_selec)]
			event_level = event_level[np.bitwise_and(id_selec,Iso_selec)]	

	        #Drop any events with no muons after selection
			tau = tau[ak.num(muon,axis=1)>0]
			boostedtau = boostedtau[ak.num(muon,axis=1)>0]
			AK8Jet = AK8Jet[ak.num(muon,axis=1)>0]
			Jet = Jet[ak.num(muon,axis=1)>0]
			electron = electron[ak.num(muon,axis=1)>0]
			muon = muon[ak.num(muon,axis=1)>0]
			event_level = event_level[ak.num(muon,axis=1)>0]

			#Fill post trigger entries in skim and N-1 histograms
			n_Trigger = np.size(event_level.nFatJet)
			h_CutFlow.fill("Trigger",weight=n_Trigger)
			h_NMinus1.fill("Trigger",weight=n_PreTrigger - n_Trigger)

		#Fill histograms after to trigger and all selections
		#Boosted Taus
		h_boostedtau_pT_Trigger.fill(ak.ravel(boostedtau.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(boostedtau.pt))[0]))
		
		if (self.nTau_Selec >= 1):
			h_Leadingboostedtau_pT_Trigger.fill(ak.ravel(boostedtau[ak.num(boostedtau,axis=1) >= self.nTau_Selec][:,0].pt),weight=ak.ravel(event_level[ak.num(boostedtau,axis=1) >= self.nTau_Selec].event_weight*CrossSec_Weight))
		
		if (self.nTau_Selec >= 2):
			h_Subleadingboostedtau_pT_Trigger.fill(ak.ravel(boostedtau[ak.num(boostedtau,axis=1) >= self.nTau_Selec][:,1].pt),weight=ak.ravel(event_level[ak.num(boostedtau,axis=1) >= self.nTau_Selec].event_weight*CrossSec_Weight))
		
		if (self.nTau_Selec >= 3):
			h_Thirdleadingboostedtau_pT_Trigger.fill(ak.ravel(boostedtau[ak.num(boostedtau,axis=1) >= self.nTau_Selec][:,2].pt),weight=ak.ravel(event_level[ak.num(boostedtau,axis=1) >= self.nTau_Selec].event_weight*CrossSec_Weight))
		
		if (self.nTau_Selec >= 4):
			h_Fourthleadingboostedtau_pT_Trigger.fill(ak.ravel(boostedtau[ak.num(boostedtau,axis=1) >= self.nTau_Selec][:,3].pt),weight=ak.ravel(event_level[ak.num(boostedtau,axis=1) >= self.nTau_Selec].event_weight*CrossSec_Weight))
		
		h_boostedtau_eta_Trigger.fill(ak.ravel(boostedtau.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(boostedtau.eta))[0]))
		h_boostedtau_phi_Trigger.fill(ak.ravel(boostedtau.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(boostedtau.phi))[0]))
		h_boostedtau_raw_iso_Trigger.fill(ak.ravel(boostedtau.iso),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(boostedtau.iso))[0]))

		#HPS Taus
		h_tau_pT_Trigger.fill(ak.ravel(tau.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(tau.pt))[0]))
		h_Leadingtau_pT_Trigger.fill(ak.ravel(tau[ak.num(tau,axis=1) > 0][:,0].pt),weight=ak.ravel(event_level[ak.num(tau,axis=1) > 0].event_weight*CrossSec_Weight))
		h_tau_eta_Trigger.fill(ak.ravel(tau.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(tau.eta))[0]))
		h_tau_phi_Trigger.fill(ak.ravel(tau.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(tau.phi))[0]))
		
		#Electrons
		h_electron_pT_Trigger.fill(ak.ravel(electron.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(electron.pt))[0]))
		h_Leadingelectron_pT_Trigger.fill(ak.ravel(electron[ak.num(electron,axis=1) > 0][:,0].pt),weight=ak.ravel(event_level[ak.num(electron,axis=1) > 0].event_weight*CrossSec_Weight))
		h_electron_eta_Trigger.fill(ak.ravel(electron.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(electron.eta))[0]))
		h_electron_phi_Trigger.fill(ak.ravel(electron.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(electron.phi))[0]))

		#Muons
		h_muon_pT_Trigger.fill(ak.ravel(muon.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(muon.pt))[0]))
		h_Leadingmuon_pT_Trigger.fill(ak.ravel(muon[ak.num(muon.pt,axis=1) > 0][:,0].pt),weight=ak.ravel(event_level[ak.num(muon.pt,axis=1) > 0].event_weight*CrossSec_Weight))
		h_muon_eta_Trigger.fill(ak.ravel(muon.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(muon.eta))[0]))
		h_Leadingmuon_eta_Trigger.fill(ak.ravel(muon[ak.num(muon.pt,axis=1) > 0][:,0].eta),weight=ak.ravel(event_level[ak.num(muon.pt,axis=1) > 0].event_weight*CrossSec_Weight))
		h_muon_phi_Trigger.fill(ak.ravel(muon.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(muon.phi))[0]))

		#Jets 
		h_Jet_pT_Trigger.fill(ak.ravel(Jet.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(Jet.pt))[0]))
		h_LeadingJet_pT_Trigger.fill(ak.ravel(Jet[ak.num(Jet,axis=1) > 0][:,0].pt),weight=ak.ravel(event_level[ak.num(Jet,axis=1) > 0].event_weight*CrossSec_Weight))
		h_Jet_eta_Trigger.fill(ak.ravel(Jet.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(Jet.eta))[0]))
		h_Jet_phi_Trigger.fill(ak.ravel(Jet.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(Jet.phi))[0]))
		
		#AK8/Fat Jets
		h_AK8Jet_pT_Trigger.fill(ak.ravel(AK8Jet.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet.pt))[0]))
		h_LeadingAK8Jet_pT_Trigger.fill(ak.ravel(AK8Jet[ak.num(AK8Jet,axis=1) > 0][:,0].pt),weight=ak.ravel(event_level[ak.num(AK8Jet,axis=1) > 0].event_weight*CrossSec_Weight))
		h_AK8Jet_eta_Trigger.fill(ak.ravel(AK8Jet.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet.eta))[0]))
		h_AK8Jet_phi_Trigger.fill(ak.ravel(AK8Jet.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet.phi))[0]))

		#Store MET
		h_MET_Trigger.fill(ak.ravel(event_level.MET_pt),weight=ak.ravel(event_level.event_weight*CrossSec_Weight))
		h_HT_Trigger.fill(ak.ravel(event_level.HT),weight=ak.ravel(event_level.event_weight*CrossSec_Weight))
		h_MHT_Trigger.fill(ak.ravel(event_level.MHT),weight=ak.ravel(event_level.event_weight*CrossSec_Weight))
		
		return{
			dataset: {
				#"Weight": CrossSec_Weight,
				"Weight_Val": CrossSec_Weight,
				"Weight": ak.to_list(event_level.event_weight*CrossSec_Weight), 
				"Event_Count": np.sum(ak.to_list(event_level.event_weight*CrossSec_Weight)),
				"n_Skim": n_Skim,
				"n_MET": n_MET,
				"n_FatJet": n_FatJet,
				"n_FlagSelec": n_FlagSelec,
				"n_PVSelec": n_PVSelec,
				"n_LeadTau": n_LeadTau,
				"n_SubLeadTau": n_SubLeadTau,
				"n_3rdLeadTau": n_3rdLeadTau,
				"n_4thLeadTau": n_4thLeadTau,
				
				#Boosted Tau kineamtic distirubtions
				"boostedtau_pt_Trigg": h_boostedtau_pT_Trigger,
				"Leadingboostedtau_pt_Trigg": h_Leadingboostedtau_pT_Trigger,
				"Subleadingboostedtau_pt_Trigg": h_Subleadingboostedtau_pT_Trigger,
				"Thirdleadingboostedtau_pt_Trigg": h_Thirdleadingboostedtau_pT_Trigger,
				"Fourthleadingboostedtau_pt_Trigg": h_Fourthleadingboostedtau_pT_Trigger,
				"boostedtau_eta_Trigg": h_boostedtau_eta_Trigger,
				"boostedtau_phi_Trigg": h_boostedtau_phi_Trigger,
				"boostedtau_iso_Trigg": h_boostedtau_raw_iso_Trigger,
				
				#HPS Tau kineamtic distirubtions
				"tau_pt_Trigg": h_tau_pT_Trigger,
				"Leadingtau_pt_Trigg": h_Leadingtau_pT_Trigger,
				"tau_eta_Trigg": h_tau_eta_Trigger,
				"tau_phi_Trigg": h_tau_phi_Trigger,
				
				#Electron kineamtic distirubtions
				"electron_pt_Trigg": h_electron_pT_Trigger,
				"Leadingelectron_pt_Trigg": h_Leadingelectron_pT_Trigger,
				"electron_eta_Trigg": h_electron_eta_Trigger,
				"electron_phi_Trigg": h_electron_phi_Trigger,
				
				#Muon kineamtic distirubtions
				"muon_pt_Trigg": h_muon_pT_Trigger,
				"Leadingmuon_pt_Trigg": h_Leadingmuon_pT_Trigger,
				"muon_eta_Trigg": h_muon_eta_Trigger,
                "Leadingmuon_eta_Trigg": h_Leadingmuon_eta_Trigger,
				"muon_phi_Trigg": h_muon_phi_Trigger,
				
				#Jet kineamtic distirubtions
				"Jet_pt_Trigg": h_Jet_pT_Trigger,
				"LeadingJet_pt_Trigg": h_LeadingJet_pT_Trigger,
				"Jet_eta_Trigg": h_Jet_eta_Trigger,
				"Jet_phi_Trigg": h_Jet_phi_Trigger,
				
				#AK8Jet kineamtic distirubtions
				"AK8Jet_pt_Trigg": h_AK8Jet_pT_Trigger,
				"LeadingAK8Jet_pt_Trigg": h_LeadingAK8Jet_pT_Trigger,
				"AK8Jet_eta_Trigg": h_AK8Jet_eta_Trigger,
				"AK8Jet_phi_Trigg": h_AK8Jet_phi_Trigger,
				
				#Print MET
				"MET": h_MET_Trigger,
				"HT": h_HT_Trigger,
				"MHT": h_MHT_Trigger,

				#Store the Mini Cutflow and N-1 Table 
				"Mini_Cutflow": h_CutFlow,
				"Mini_NMinus1": h_NMinus1,
			}
		}

	def postprocess(self, accumulator):
		pass

if __name__ == "__main__":
	#Condor related stuff
	os.environ["CONDOR_CONFIG"] = "/etc/condor/condor_config"
	
	#Xrootd crap
	_x509_path = move_X509()
	print(f"x509 path: {_x509_path}")
	htc_log_err_dir = "/scratch/twnelson/ControlPlot_HTC/Run_" + str(time.localtime()[0]) + "_" + str(time.localtime()[1]) + "_" + str(time.localtime()[2]) + "_" + str(time.localtime()[3]) + f".{time.localtime()[4]:02d}"
	os.makedirs(htc_log_err_dir)

	cluster = HTCondorCluster(
			cores=1,
			memory="4 GB",
			disk="2.0 GB",
			death_timeout = '60',
			job_extra_directives={
				"+JobFlavour": '"tomorrow"',
				"log": "dask_job_output.$(PROCESS).$(CLUSTER).log",
				"output": "dask_job_output.$(PROCESS).$(CLUSTER).out",
				"error": "dask_job_output.$(PROCESS).$(CLUSTER).err",
				"should_transfer_files": "yes",
				"when_to_transfer_ouput": "ON_EXIT_OR_EVICT",
				"transfer_executable": "false",
				"+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest-py3.10"',
				"Requirements": "HasSingularityJobStart",
				"InitialDir": f'/scratch/{os.environ["USER"]}',
				'transfer_input_files': f"{_x509_path}",

			},
			job_script_prologue = [
				"export XRD_RUNFORKHANDLER=1",
				f"export X509_USER_PROXY={_x509_path}",
			]
	)
	cluster.adapt(minimum=1, maximum=500)

	run_on_condor = True 
	
	if (run_on_condor):
		print("Run on Condor")
		runner = processor.Runner(
			executor = processor.DaskExecutor(client=Client(cluster),status=False),
			schema=BaseSchema,
			skipbadfiles=True,
			xrootdtimeout=1000,
            #chunksize=500000,
            #maxchunks = 1
		)
	else: #Iterative runner
		runner = processor.Runner(executor = processor.IterativeExecutor(), schema=BaseSchema)
	
	four_tau_hist_list = [
			"boostedtau_pt_Trigg","Leadingboostedtau_pt_Trigg", "Subleadingboostedtau_pt_Trigg", "Thirdleadingboostedtau_pt_Trigg","Fourthleadingboostedtau_pt_Trigg",
			"boostedtau_eta_Trigg","boostedtau_phi_Trigg",
			#"boostedtau_pt_Trigg","Leadingboostedtau_pt_Trigg","boostedtau_eta_Trigg","boostedtau_phi_Trigg",
			"tau_pt_Trigg","Leadingtau_pt_Trigg","tau_eta_Trigg","tau_phi_Trigg",
			"electron_pt_Trigg","Leadingelectron_pt_Trigg","electron_eta_Trigg","electron_phi_Trigg",
			"muon_pt_Trigg","Leadingmuon_pt_Trigg","muon_eta_Trigg","muon_phi_Trigg",
			"Jet_pt_Trigg","LeadingJet_pt_Trigg","Jet_eta_Trigg","Jet_phi_Trigg",
			"AK8Jet_pt_Trigg","LeadingAK8Jet_pt_Trigg","AK8Jet_eta_Trigg","AK8Jet_phi_Trigg",
			"MET","HT","MHT",
			] 

	hist_name_dict = {
					"boostedtau_pt_Trigg": r"Boosted $\tau$ $p_T$ after Trigger"," Leadingboostedtau_pt_Trigg": r"Leading boosted $\tau$ $p_T$ after Trigger",
					"Subleadingboostedtau_pt_Trigg": r"Subleading boosted $\tau$ $p_T$ after Trigger",
					"Thirdleadingboostedtau_pt_Trigg": r"3rd leading boosted $\tau$ $p_T$ after Trigger",
					"Fourthleadingboostedtau_pt_Trigg": r"4th leading boosted $\tau$ $p_T$ after Trigger",
					"boostedtau_eta_Trigg": r"Boosted $\tau$ $\eta$ after Trigger","boostedtau_phi_Trigg": r"Boosted $\tau$ $\phi$ after Trigger", 
					"tau_pt_Trigg": r"$\tau$ $p_T$ after Trigger","Leadingtau_pt_Trigg": r"Leading $\tau$ $p_T$ after Trigger",
					"tau_eta_Trigg": r"$\tau$ $\eta$ after Trigger","tau_phi_Trigg": r"$\tau$ $\phi$ after Trigger", 
					"electron_pt_Trigg": r"e $p_T$ after Trigger","Leadingelectron_pt_Trigg": r"Leading e $p_T$ after Trigger",
					"electron_eta_Trigg": r"e $\eta$ after Trigger", "electron_phi_Trigg": r"e $\phi$ after Trigger", 
					"muon_pt_Trigg": r"$\mu$ $p_T$ after Trigger","Leadingmuon_pt_Trigg": r"Leading $\mu$ $p_T$ after Trigger",
					"muon_eta_Trigg": r"$\mu$ $\eta$ after Trigger","muon_phi_Trigg": r"$\mu$ $\phi$ after Trigger", 
					"Jet_pt_Trigg": r"Jet $p_T$ after Trigger","LeadingJet_pt_Trigg": r"LeadingJet $p_T$ after Trigger",
					"Jet_eta_Trigg": r"Jet $\eta$ after Trigger", "Jet_phi_Trigg": r"Jet $\phi$ after Trigger", 
					"AK8Jet_pt_Trigg": r"AK8Jet $p_T$ after Trigger","LeadingAK8Jet_pt_Trigg": r"Leading AK8Jet $p_T$ after Trigger",
					"AK8Jet_eta_Trigg": r"AK8Jet $\eta$ after Trigger","AK8Jet_phi_Trigg": r"AK8Jet $\phi$ after Trigger",
					"MET": r"MET after Trigger", "HT": r"HT after Trigger", "MHT": r"MHT after Trigger",
					}

	#Diretory for files
	Skimmed_4tau_base_MC = "root://cmsxrootd.hep.wisc.edu//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/"
	Skimmed_4tau_base_Data = "root://cmsxrootd.hep.wisc.edu//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/Data/"
	Skimmed_4tau_loc_Data = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/Data/"
	Skimmed_4tau_loc_MC = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/"

	#Make full arrays of single Muon data
	SingleMuA_2018A = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 
	SingleMuA_2018B = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018B_15January26_0731_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 
	SingleMuA_2018C = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018C_15January26_0740_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 
	SingleMuA_2018D = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018D_15January26_0815_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 

	#Make full arrays of backgrounds
	TTToSemiLeptonic_2018 = glob.glob(Skimmed_4tau_loc_MC + "TTToSemiLeptonic_35August25_0448_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	TTTo2L2Nu_2018 = glob.glob(Skimmed_4tau_loc_MC + "TTTo2L2Nu_26August25_0719_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	TTToHadronic_2018 = glob.glob(Skimmed_4tau_loc_MC + "TTToHadronic_25October25_0813_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZ4L_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo4L_26August25_0757_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZTo2L2Nu_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo2L2Nu_04March26_0503_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZTo2L2Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo2Q2L_26August25_1034_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZTo2Nu2Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo2Nu2Q_04March26_0510_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZTo4Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo4Q_04March26_0505_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	VV2l2nu_2018 = glob.glob(Skimmed_4tau_loc_MC + "WWTo2L2Nu_26August25_1040_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WWTo1L1Nu2Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "WWTo2L2Nu_26August25_1040_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WWTo4Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "WWTo4Q_04March26_0512_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WZ1l3nu_2018 = glob.glob(Skimmed_4tau_loc_MC + "WZTo1L3Nu_4f_26August25_1016_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZ2l2q_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo2Q2L_26August25_1034_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WZ2l2q_2018 = glob.glob(Skimmed_4tau_loc_MC + "WZTo2L2Q_26August25_0926_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WZ1l1nu2q_2018 = glob.glob(Skimmed_4tau_loc_MC + "WZTo1L1Nu2Q_26August25_0840_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT70to100_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-70to100_12December25_1606_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT100to200_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-100to200_12December25_1604_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT200to400_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-200to400_12December25_1544_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT400to600_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-400to600_12December25_1552_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT600toInf_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-600toInf_12December25_1608_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT70to100_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-70to100_12December25_1556_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT100to200_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-100to200_12December25_1548_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT200to400_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-200to400_12December25_1559_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT400to600_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-400to600_12December25_1546_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT600to800_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-600to800_12December25_1555_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT800to1200_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-800to1200_12December25_1602_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT1200to2500_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-1200to2500_12December25_1547_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT2500toInf_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-1200to2500_12December25_1547_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	Ttchan_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_t-channel_top_4f_InclusiveDecays_26August25_0843_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	Tbartchan_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_t-channel_antitop_4f_InclusiveDecays_26August25_0821_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	TtW_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_tW_top_5f_inclusiveDecays_26August25_0753_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	TbartW_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_tW_antitop_5f_inclusiveDecays_26August25_1030_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ST_schannel_4f_hadronicDecays_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_s-channel_4f_hadronicDecays_04March26_0506_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ST_schannel_4f_leptonDecays_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_s-channel_4f_leptonDecays_04March26_0507_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WJetsToLNu_HT70To100_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-70To100_04March26_0515_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WJetsToLNu_HT100To200_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-100To200_26August25_0810_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WJetsToLNu_HT200To400_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-200To400_26August25_0709_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WJetsToLNu_HT400To600_2018 = np.append(glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-400To600_26August25_1014_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-400To600_OtherPart_26August25_1032_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	WJetsToLNu_HT600To800_2018 = np.append(glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-600To800_26August25_0755_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-600To800_OtherPart_26August25_0752_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	WJetsToLNu_HT800To1200_2018 = np.append(glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-800To1200_26August25_0708_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-800To1200_OtherPart_26August25_0925_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	WJetsToLNu_HT1200To2500_2018 = np.append(glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-120)0To2500_26August25_1016_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-1200To2500_OtherPart_26August25_1041_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	WJetsToLNu_HT2500ToInf_2018 = np.append(glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-2500ToInf_26August25_1047_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-2500ToInf_OtherPart_26August25_1043_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))

	file_dict_data_test = {
		#"Data_Mu" : [Skimmed_4tau_base_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/SingleMu_Run2018A.root"]
		#"Data_Mu": [Skimmed_4tau_base_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root"]
		"Data_Mu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in SingleMuA_2018A] 
	}
	
	file_dict_full = {
			"TTToSemiLeptonic": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TTToSemiLeptonic_2018],
			"TTTo2L2Nu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TTTo2L2Nu_2018],
			"TTToHadronic": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TTToHadronic_2018],
			"ZZ4l": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZ4L_2018],
			"ZZTo2L2Nu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZTo2L2Nu_2018],
			"ZZTo2Nu2Q": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZTo2Nu2Q_2018],
			"VV2l2nu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in VV2l2nu_2018],
			"ZZTo4Q" : ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZTo4Q_2018],
			"WWTo1L1Nu2Q": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WWTo1L1Nu2Q_2018],
			"WWTo4Q": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WWTo4Q_2018],
			"WZ1l3nu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WZ1l3nu_2018],
			"ZZ2l2q": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZ2l2q_2018],
			"WZ2l2q": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WZ2l2q_2018],
			"WZ1l1nu2q" : ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WZ1l1nu2q_2018],
			"DYJetsToLL_M-4to50_HT-70to100": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M4to50_HT70to100_2018],
			"DYJetsToLL_M-4to50_HT-100to200": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M4to50_HT100to200_2018],
			"DYJetsToLL_M-4to50_HT-200to400": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M4to50_HT200to400_2018],
			"DYJetsToLL_M-4to50_HT-400to600": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M4to50_HT400to600_2018],
			"DYJetsToLL_M-4to50_HT-600toInf":["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M4to50_HT600toInf_2018],
			"DYJetsToLL_M-50_HT-70to100": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT70to100_2018],
			"DYJetsToLL_M-50_HT-100to200": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT100to200_2018],
			"DYJetsToLL_M-50_HT-200to400": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT200to400_2018],
			"DYJetsToLL_M-50_HT-400to600": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT400to600_2018],
			"DYJetsToLL_M-50_HT-600to800": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT600to800_2018],
			"DYJetsToLL_M-50_HT-800to1200": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT800to1200_2018],
			"DYJetsToLL_M-50_HT-1200to2500": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT1200to2500_2018],
			"DYJetsToLL_M-50_HT-2500toInf": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT2500toInf_2018],
			"T-tchan": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in Ttchan_2018],
			"Tbar-tchan": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in Tbartchan_2018],
			"T-tW": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TtW_2018],
			"Tbar-tW": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TbartW_2018],
			"ST_s-channel_4f_hadronicDecays": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ST_schannel_4f_hadronicDecays_2018],
			"ST_s-channel_4f_leptonDecays": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ST_schannel_4f_leptonDecays_2018],
			"WJetsToLNu_HT-70To100": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT70To100_2018],
			"WJetsToLNu_HT-100To200": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT100To200_2018],
			"WJetsToLNu_HT-200To400": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT200To400_2018],
			"WJetsToLNu_HT-400To600": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT400To600_2018],
			"WJetsToLNu_HT-600To800": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT600To800_2018],
			"WJetsToLNu_HT-800To1200": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT800To1200_2018],
			"WJetsToLNu_HT-1200To2500": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT1200To2500_2018],
			"WJetsToLNu_HT-2500ToInf": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT2500ToInf_2018],
			"Data_Mu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in np.append(SingleMuA_2018A, np.append(SingleMuA_2018B, np.append(SingleMuA_2018C,SingleMuA_2018D)))]
		}
	
	#Background lists 
	background_list_full = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$","QCD"] #Full background list (with QCD)
	background_list_full = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"] #Full background list
	background_list_test = [r"$ZZ \rightarrow 4l$"] #Only ZZ4l background for testing
	background_list_none = [] #No backgrounds for data only testing
	
	#Set file dictionary and list of backgrounds prior to running processor
	#file_dict = file_dict_data_test
	file_dict = file_dict_full

	#Pull in the weight and event count prior to skimming information
	with open("genWeightSum_JSON.json") as json_file:
		sumWEvents_Dict = json.load(json_file)

	with open("numEvents_JSON.json") as json_file:
		numEvents_Dict = json.load(json_file)
	

	start_time = time.time()
#	for key_name, file_array in file_dict.items(): 
#		print(key_name)
#		if (key_name != "Data_Mu" and key_name != "Data_MET" ): 
#			numEvents_Dict[key_name] = 0 #Initialize the number of events dictionary
#			sumWEvents_Dict[key_name] = 0 #Initialize the number of events dictionary
#			for file in file_array:
#				#with uproot.open(file) as tempFile:
#				#print(file)
#				tempFile = uproot.dask(file + ":Runs")
#				numEvents_Dict[key_name] += np.sum(tempFile['genEventCount'].compute()[0]) #Fixed for nanoAOD 
#				sumWEvents_Dict[key_name] += np.sum(tempFile['genEventSumw'].compute()[0]) #Fixed for nanoAOD 
#				del tempFile
#			
#			#	numEvents_Dict[key_name] += np.sum(tempFile['Runs/genEventCount'].array()) #Fixed for nanoAOD 
#			#	sumWEvents_Dict[key_name] += np.sum(tempFile['Runs/genEventSumw'].array()) #Fixed for nanoAOD 
#			#	print(key_name + "sum: %f"%numEvents_Dict[key_name])
#
#		else: #Ignore data files
#			numEvents_Dict[key_name] = 1
#			sumWEvents_Dict[key_name] = 1
	
	end_time_0 = time.time()
	print("Time to process all files: %d"%(end_time_0 - start_time))

	#Background names for single background plot file names
	background_plot_names = {r"$t\bar{t}$" : "_ttbar_", r"$t\bar{t}$ Hadronic" : "_ttbarHadronic_", r"$t\bar{t}$ Semileptonic" : "_ttbarSemilepton_",
			r"$t\bar{t}$ 2L2Nu" : "_ttbar2L2Nu_", r"Drell-Yan+Jets": "_DYJets_", "Di-Bosons" : "_DiBosons_", "Single Top": "_SingleTop_", "QCD" : "_QCD_", 
			"W+Jets" : "_WJets_", r"$ZZ \rightarrow 4l$" : "_ZZ4l_", r"$ZZ \rightarrow 4l$ Test": "_ZZ4lTest_", r"$ZZ \rightarrow 4l$ Control": "_ZZ4lControl_",
			"W+Jets HT 70-100 GeV" : "_WJetsHT70-100_","W+Jets HT 100-200 GeV" : "_WJetsHT100-200_","W+Jets HT 200-400 GeV" : "_WJetsHT200-400_",
			"W+Jets HT 400-600 GeV" : "_WJetsHT400-600_","W+Jets HT 600-800 GeV" : "_WJetsHT600-800_","W+Jets HT 800-1200 GeV" : "_WJetsHT800-1200_",
			"W+Jets HT 1200-2500 GeV" : "_WJetsHT1200-2500_","W+Jets HT 2500-Inf GeV" : "_WJetsHT2500-Inf_"} #For file names
	
	#Background names to samples dictionary
	background_dict = {r"$t\bar{t}$" : ["TTToSemiLeptonic","TTTo2L2Nu","TTToHadronic"], r"$t\bar{t}$ Hadronic" : ["TTToHadronic"], 
			r"$t\bar{t}$ Semileptonic" : ["TTToSemiLeptonic"], r"$t\bar{t}$ 2L2Nu" : ["TTTo2L2Nu"],
			r"Drell-Yan+Jets": ["DYJetsToLL_M-4to50_HT-70to100","DYJetsToLL_M-4to50_HT-100to200","DYJetsToLL_M-4to50_HT-200to400","DYJetsToLL_M-4to50_HT-400to600",
			"DYJetsToLL_M-4to50_HT-600toInf","DYJetsToLL_M-50_HT-70to100","DYJetsToLL_M-50_HT-100to200","DYJetsToLL_M-50_HT-200to400",
			"DYJetsToLL_M-50_HT-400to600","DYJetsToLL_M-50_HT-600to800","DYJetsToLL_M-50_HT-800to1200","DYJetsToLL_M-50_HT-1200to2500","DYJetsToLL_M-50_HT-2500toInf"], 
			"Di-Bosons": ["WZ2l2q","WZ1l1nu2q","ZZ2l2q", "WZ1l3nu", "VV2l2nu", "WWTo1L1Nu2Q", "WWTo4Q", "ZZTo4Q", "ZZTo2L2Nu", "ZZTo2Nu2Q"], 
			"Single Top": ["Tbar-tchan","T-tchan","Tbar-tW","T-tW","ST_s-channel_4f_leptonDecays", "ST_s-channel_4f_hadronicDecays"], 
			"W+Jets": ["WJetsToLNu_HT-70To100","WJetsToLNu_HT-100To200","WJetsToLNu_HT-200To400","WJetsToLNu_HT-400To600","WJetsToLNu_HT-600To800","WJetsToLNu_HT-800To1200","WJetsToLNu_HT-1200To2500","WJetsToLNu_HT-2500ToInf"],
			"W+Jets HT 100-200 GeV": ["WJetsToLNu_HT-100To200"],"W+Jets HT 200-400 GeV": ["WJetsToLNu_HT-200To400"],"W+Jets HT 400-600 GeV": ["WJetsToLNu_HT-400To600"],
			"W+Jets HT 600-800 GeV": ["WJetsToLNu_HT-600To800"],"W+Jets HT 800-1200 GeV": ["WJetsToLNu_HT-800To1200"],
			"W+Jets HT 1200-2500 GeV": ["WJetsToLNu_HT-1200To2500"], "W+Jets HT 2500-Inf GeV": ["WJetsToLNu_HT-2500ToInf"],
			r"$ZZ \rightarrow 4l$" : ["ZZ4l"],
			"QCD": ["QCD_HT50to100","QCD_HT100to200","QCD_HT200to300","QCD_HT300to500","QCD_HT500to700","QCD_HT700to1000","QCD_HT1000to1500","QCD_HT1500to2000","QCD_HT2000toInf"],
	}

	#Dictinary with file names
	trigger_name = "Mu_Trigger_WithQCD"
	four_tau_names = {
		"boostedtau_pt_Trigg": "BoostedTau_pT_Trigger" + "-" + trigger_name,
		"Leadingboostedtau_pt_Trigg": "BoostedTau_Leading_pT_Trigger" + "-" + trigger_name,
		"Subleadingboostedtau_pt_Trigg": "BoostedTau_Subleading_pT_Trigger" + "-" + trigger_name, 
		"Thirdleadingboostedtau_pt_Trigg": "BoostedTau_3rdleading_pT_Trigger" + "-" + trigger_name, 
		"Fourthleadingboostedtau_pt_Trigg": "BoostedTau_4thleading_pT_Trigger" + "-" + trigger_name, 
		"boostedtau_eta_Trigg": "BoostedTau_eta_Trigger" + "-" + trigger_name,
		"boostedtau_phi_Trigg": "BoostedTau_phi_Trigger" + "-" + trigger_name,
		"boostedtau_iso_Trigg": "BoostedTau_iso_Trigger" + "-" + trigger_name,
		"tau_pt_Trigg": "Tau_pT_Trigger" + "-" + trigger_name,
		"Leadingtau_pt_Trigg": "Tau_Leading_pT_Trigger" + "-" + trigger_name,
		"tau_eta_Trigg": "Tau_eta_Trigger" + "-" + trigger_name,
		"tau_phi_Trigg": "Tau_phi_Trigger" + "-" + trigger_name,
		"electron_pt_Trigg": "Electron_pT_Trigger" + "-" + trigger_name,
		"Leadingelectron_pt_Trigg": "Electron_Leading_pT_Trigger" + "-" + trigger_name,
		"electron_eta_Trigg": "Electron_eta_Trigger" + "-" + trigger_name,
		"electron_phi_Trigg": "Electron_phi_Trigger" + "-" + trigger_name,
		"muon_pt_Trigg": "Muon_pT_Trigger" + "-" + trigger_name,
		"Leadingmuon_pt_Trigg": "Muon_Leading_pT_Trigger" + "-" + trigger_name,
		"muon_eta_Trigg": "Muon_eta_Trigger" + "-" + trigger_name,
		"muon_phi_Trigg": "Muon_phi_Trigger" + "-" + trigger_name,
		"Jet_pt_Trigg": "Jet_pT_Trigger" + "-" + trigger_name,
		"LeadingJet_pt_Trigg": "Jet_Leading_pT_Trigger" + "-" + trigger_name,
		"Jet_eta_Trigg": "Jet_eta_Trigger" + "-" + trigger_name,
		"Jet_phi_Trigg": "Jet_phi_Trigger" + "-" + trigger_name,
		"AK8Jet_pt_Trigg": "AK8Jet_pT_Trigger" + "-" + trigger_name,
		"LeadingAK8Jet_pt_Trigg": "AK8Jet_Leading_pT_Trigger" + "-" + trigger_name,
		"AK8Jet_eta_Trigg": "AK8Jet_eta_Trigger" + "-" + trigger_name,
		"AK8Jet_phi_Trigg": "AK8Jet_phi_Trigger" + "-" + trigger_name,
		"MET": "MET_Trigger" + "-" + trigger_name,
		"HT": "HT_Trigger" + "-" + trigger_name,
		"MHT": "MHT_Trigger" + "-" + trigger_name,
	}
	
	print(os.getcwd())
	for n_taus in range(4,5):
		#print(os.getcwd())
		print("About to run processor")
		start_time = time.time()
		fourtau_out = runner(file_dict, treename="Events", processor_instance=PlottingScriptProcessor(nBoostedTaus = n_taus, ApplyTrigger = False)) #Modified for NanoAOD (changd treename)
		end_time = time.time()
		
		time_running = end_time-start_time
		print("It takes about %.1f s to run the coffea processor with %d boosted tau selections"%(time_running,n_taus))
		
        #Save coffea file
		outfile = os.path.join(os.getcwd(), f"output_{n_taus}_boosted_tau_selec_SingleMu2018A.coffea")
		#outfile = "~/Analysis/BoostedTau/ControlPlots/DebuggingStudies/AnalysisCompDebugging/Studies_4tau/SimpleSelec_2b2tauSamples/QCD_Studies/" + f"output_{n_taus}_boosted_tau_selec.coffea"
		#timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		#outfile = os.path.join(os.getcwd(), f"output_2018_run{timestamp}.coffea")
		util.save(fourtau_out, outfile)
		print(f"Saved output to {outfile}")	
