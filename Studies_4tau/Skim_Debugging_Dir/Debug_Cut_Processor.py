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
	def __init__(self, nBoostedTaus = 0, ApplyTrigger = True, DebugCode = 8): #Additional arguements can be added later
		self.isData = False #Default assumption is MC
		self.nTau_Selec = nBoostedTaus #Number of tau selections
		self.ApplyTrigger = ApplyTrigger
		self.Debug_Code = DebugCode
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
		
		if (self.Debug_Code >= 1):
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
		if (self.Debug_Code >= 2):
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
		if (self.Debug_Code >= 3):
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
		if (self.Debug_Code >= 4):
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
		if (self.nTau_Selec > 0 and self.Debug_Code >= 5):
			#Require events to have at least n boosted tau
			tau = tau[ak.num(boostedtau,axis=1) >= self.nTau_Selec]
			boostedtau = boostedtau[ak.num(boostedtau,axis=1) >= self.nTau_Selec]
			AK8Jet = AK8Jet[ak.num(boostedtau,axis=1) >= self.nTau_Selec]
			Jet = Jet[ak.num(boostedtau,axis=1) >= self.nTau_Selec]
			electron = electron[ak.num(boostedtau,axis=1) >= self.nTau_Selec]
			muon = muon[ak.num(boostedtau,axis=1) >= self.nTau_Selec]
			event_level = event_level[ak.num(boostedtau,axis=1) >= self.nTau_Selec]

			#Impose selections on leading boosted tau
			pT_Cond = boostedtau[:,0].pt > 20
			eta_Cond = np.abs(boostedtau[:,0].eta) < 2.3
			decayMode_Cond = boostedtau[:,0].decay >= 0.5
			DBT_Iso_Mode_Cond = boostedtau[:,0].DBT >= 0.5

			tau_lead_selec = np.bitwise_and(DBT_Iso_Mode_Cond,np.bitwise_and(decayMode_Cond,np.bitwise_and(pT_Cond,eta_Cond)))
			
			tau = tau[tau_lead_selec]
			boostedtau = boostedtau[tau_lead_selec]
			AK8Jet = AK8Jet[tau_lead_selec]
			Jet = Jet[tau_lead_selec]
			electron = electron[tau_lead_selec]
			muon = muon[tau_lead_selec]
			event_level = event_level[tau_lead_selec]

			#Fill post leading tau selection entries in skim and N-1 histograms
			n_LeadTau = np.size(event_level.nFatJet)
			h_CutFlow.fill("LeadingTau",weight=n_LeadTau)
			h_NMinus1.fill("LeadingTau",weight=n_PVSelec - n_LeadTau)

			n_PreTrigger = n_LeadTau				
			
			#Impose selections on Subleading boosted tau
			if (self.nTau_Selec > 1 and self.Debug_Code >= 6):
				pT_Cond = boostedtau[:,1].pt > 20
				eta_Cond = np.abs(boostedtau[:,1].eta) < 2.3
				decayMode_Cond = boostedtau[:,1].decay >= 0.5
				DBT_Iso_Mode_Cond = boostedtau[:,1].DBT >= 0.5

				tau_sublead_selec = np.bitwise_and(DBT_Iso_Mode_Cond,np.bitwise_and(decayMode_Cond,np.bitwise_and(pT_Cond,eta_Cond)))
				
				tau = tau[tau_sublead_selec]
				boostedtau = boostedtau[tau_sublead_selec]
				AK8Jet = AK8Jet[tau_sublead_selec]
				Jet = Jet[tau_sublead_selec]
				electron = electron[tau_sublead_selec]
				muon = muon[tau_sublead_selec]
				event_level = event_level[tau_sublead_selec]

				#Fill post subl-leading tau selection entries in skim and N-1 histograms
				n_SubLeadTau = np.size(event_level.nFatJet)
				h_CutFlow.fill("SubleadingTau",weight=n_SubLeadTau)
				h_NMinus1.fill("SubleadingTau",weight=n_LeadTau - n_SubLeadTau)

				n_PreTrigger = n_SubLeadTau				
			
			#Impose selections on third-leading boosted tau
			if (self.nTau_Selec > 2 and self.Debug_Code >= 7):
				pT_Cond = boostedtau[:,2].pt > 20
				eta_Cond = np.abs(boostedtau[:,2].eta) < 2.3
				decayMode_Cond = boostedtau[:,2].decay >= 0.5
				DBT_Iso_Mode_Cond = boostedtau[:,2].DBT >= 0.5

				tau_3lead_selec = np.bitwise_and(DBT_Iso_Mode_Cond,np.bitwise_and(decayMode_Cond,np.bitwise_and(pT_Cond,eta_Cond)))
				
				tau = tau[tau_3lead_selec]
				boostedtau = boostedtau[tau_3lead_selec]
				AK8Jet = AK8Jet[tau_3lead_selec]
				Jet = Jet[tau_3lead_selec]
				electron = electron[tau_3lead_selec]
				muon = muon[tau_3lead_selec]
				event_level = event_level[tau_3lead_selec]	

				#Fill post 3rd leading tau selection entries in skim and N-1 histograms
				n_3rdLeadTau = np.size(event_level.nFatJet)
				h_CutFlow.fill("3rdLeadingTau",weight=n_3rdLeadTau)
				h_NMinus1.fill("3rdLeadingTau",weight=n_SubLeadTau - n_3rdLeadTau)

				n_PreTrigger = n_3rdLeadTau				
			
			#Impose selections on fourth-leading boosted tau
			if (self.nTau_Selec > 3 and self.Debug_Code >= 8):
				pT_Cond = boostedtau[:,3].pt > 20
				eta_Cond = np.abs(boostedtau[:,3].eta) < 2.3
				decayMode_Cond = boostedtau[:,3].decay >= 0.5
				DBT_Iso_Mode_Cond = boostedtau[:,3].DBT >= 0.5

				tau_4lead_selec = np.bitwise_and(DBT_Iso_Mode_Cond,np.bitwise_and(decayMode_Cond,np.bitwise_and(pT_Cond,eta_Cond)))
				
				tau = tau[tau_4lead_selec]
				boostedtau = boostedtau[tau_4lead_selec]
				AK8Jet = AK8Jet[tau_4lead_selec]
				Jet = Jet[tau_4lead_selec]
				electron = electron[tau_4lead_selec]
				muon = muon[tau_4lead_selec]
				event_level = event_level[tau_4lead_selec]	

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
