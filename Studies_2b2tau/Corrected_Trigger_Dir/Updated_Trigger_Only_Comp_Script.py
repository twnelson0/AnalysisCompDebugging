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

	return dimass_Selec

def deltaPhi_METSelec(part1,MET,low_lim): #Note definition of deltaR coppied from coffea code (line 67 of vector.py)
	return (part1.phi - MET.pfMETPhi + np.pi) % (2*np.pi) - np.pi > low_lim

def reorder(to_reorder, template_object):
	to_reorder_cc = to_reorder[ak.num(template_object,axis=1)>0]
	to_reorder_conv = to_reorder[ak.num(template_object,axis=1) == 0]

	return ak.concatenate((to_reorder_cc,to_reorder_conv))


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
				"METHTMHT_Trigger": events.HLT_PFHT500_PFMET100_PFMHT100_IDTight,
				"Mu_Trigger": events.HLT_Mu50,
				#"MET_trigger": events.HLT_MET120_IsoTrk50,
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
				"IDSelec": events.Muon_mediumId,
				"D0": events.Muon_dxy,
				"Dz": events.Muon_dz,
				"LooseId": events.Muon_looseId,
				"RelIso": events.Muon_pfRelIso04_all,
					
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
		h_boostedtau_pT_Trigger_h = hist.Hist.new.Regular(50,0,1000,label = r"Boosted $\tau$ $p_T$ [GeV]").Double()
		h_Leadingboostedtau_pT_Trigger_h = hist.Hist.new.Regular(50,0,1000,label = r"Boosted $\tau$ $p_T$ [GeV]").Double()
		h_boostedtau_eta_Trigger_h = hist.Hist.new.Regular(20,-4,4,label = r"Boosted $\tau$ $\eta$").Double()
		h_boostedtau_phi_Trigger_h = hist.Hist.new.Regular(20,-pi,pi,label = r"Boosted$\tau$ $\phi$").Double()
		h_boostedtau_raw_iso_Trigger_h = hist.Hist.new.Regular(20,-1,1,label=r"Raw MVA Score").Double()
		
		h_boostedtau_pT_Trigger_e = hist.Hist.new.Regular(50,0,1000,label = r"Boosted $\tau$ $p_T$ [GeV]").Double()
		h_Leadingboostedtau_pT_Trigger_e = hist.Hist.new.Regular(50,0,1000,label = r"Boosted $\tau$ $p_T$ [GeV]").Double()
		h_boostedtau_eta_Trigger_e = hist.Hist.new.Regular(20,-4,4,label = r"Boosted $\tau$ $\eta$").Double()
		h_boostedtau_phi_Trigger_e = hist.Hist.new.Regular(20,-pi,pi,label = r"Boosted$\tau$ $\phi$").Double()
		h_boostedtau_raw_iso_Trigger_e = hist.Hist.new.Regular(20,-1,1,label=r"Raw MVA Score").Double()
		
		h_boostedtau_pT_Trigger_m = hist.Hist.new.Regular(50,0,1000,label = r"Boosted $\tau$ $p_T$ [GeV]").Double()
		h_Leadingboostedtau_pT_Trigger_m = hist.Hist.new.Regular(50,0,1000,label = r"Boosted $\tau$ $p_T$ [GeV]").Double()
		h_boostedtau_eta_Trigger_m = hist.Hist.new.Regular(20,-4,4,label = r"Boosted $\tau$ $\eta$").Double()
		h_boostedtau_phi_Trigger_m = hist.Hist.new.Regular(20,-pi,pi,label = r"Boosted$\tau$ $\phi$").Double()
		h_boostedtau_raw_iso_Trigger_m = hist.Hist.new.Regular(20,-1,1,label=r"Raw MVA Score").Double()
		
		#Basic Kinematic histograms of tau/HPS tau
		h_tau_pT_Trigger_h = hist.Hist.new.Regular(50,0,1000,label = r" $\tau$ $p_T$ [GeV]").Double()
		h_Leadingtau_pT_Trigger_h = hist.Hist.new.Regular(50,0,1000,label = r" $\tau$ $p_T$ [GeV]").Double()
		h_tau_eta_Trigger_h = hist.Hist.new.Regular(20,-4,4,label = r" $\tau$ $\eta$").Double()
		h_tau_phi_Trigger_h = hist.Hist.new.Regular(20,-pi,pi,label = r"$\tau$ $\phi$").Double()
		
		h_tau_pT_Trigger_e = hist.Hist.new.Regular(50,0,1000,label = r" $\tau$ $p_T$ [GeV]").Double()
		h_Leadingtau_pT_Trigger_e = hist.Hist.new.Regular(50,0,1000,label = r" $\tau$ $p_T$ [GeV]").Double()
		h_tau_eta_Trigger_e = hist.Hist.new.Regular(20,-4,4,label = r" $\tau$ $\eta$").Double()
		h_tau_phi_Trigger_e = hist.Hist.new.Regular(20,-pi,pi,label = r"$\tau$ $\phi$").Double()
		
		h_tau_pT_Trigger_m = hist.Hist.new.Regular(50,0,1000,label = r" $\tau$ $p_T$ [GeV]").Double()
		h_Leadingtau_pT_Trigger_m = hist.Hist.new.Regular(50,0,1000,label = r" $\tau$ $p_T$ [GeV]").Double()
		h_tau_eta_Trigger_m = hist.Hist.new.Regular(20,-4,4,label = r" $\tau$ $\eta$").Double()
		h_tau_phi_Trigger_m = hist.Hist.new.Regular(20,-pi,pi,label = r"$\tau$ $\phi$").Double()

		#Basic Kinematic histograms leptons (muons and electrons)
		h_electron_pT_Trigger_h = hist.Hist.new.Regular(50,0,1000,label = r"e $p_T$ [GeV]").Double()
		h_Leadingelectron_pT_Trigger_h = hist.Hist.new.Regular(50,0,1000,label = r"e $p_T$ [GeV]").Double()
		h_electron_eta_Trigger_h = hist.Hist.new.Regular(20,-4,4,label = r"e $\eta$").Double()
		h_electron_phi_Trigger_h = hist.Hist.new.Regular(20,-pi,pi,label = r"e $\phi$").Double()
		h_muon_pT_Trigger_h = hist.Hist.new.Regular(50,0,1000,label = r"$\mu$ $p_T$ [GeV]").Double()
		h_Leadingmuon_pT_Trigger_h = hist.Hist.new.Regular(50,0,1000,label = r"$\mu$ $p_T$ [GeV]").Double()
		h_muon_eta_Trigger_h = hist.Hist.new.Regular(20,-4,4,label = r"$\mu$ $\eta$").Double()
		h_muon_phi_Trigger_h = hist.Hist.new.Regular(20,-pi,pi,label = r"$\mu$ $\phi$").Double()
		
		h_electron_pT_Trigger_e = hist.Hist.new.Regular(50,0,1000,label = r"e $p_T$ [GeV]").Double()
		h_Leadingelectron_pT_Trigger_e = hist.Hist.new.Regular(50,0,1000,label = r"e $p_T$ [GeV]").Double()
		h_electron_eta_Trigger_e = hist.Hist.new.Regular(20,-4,4,label = r"e $\eta$").Double()
		h_electron_phi_Trigger_e = hist.Hist.new.Regular(20,-pi,pi,label = r"e $\phi$").Double()
		h_muon_pT_Trigger_e = hist.Hist.new.Regular(50,0,1000,label = r"$\mu$ $p_T$ [GeV]").Double()
		h_Leadingmuon_pT_Trigger_e = hist.Hist.new.Regular(50,0,1000,label = r"$\mu$ $p_T$ [GeV]").Double()
		h_muon_eta_Trigger_e = hist.Hist.new.Regular(20,-4,4,label = r"$\mu$ $\eta$").Double()
		h_muon_phi_Trigger_e = hist.Hist.new.Regular(20,-pi,pi,label = r"$\mu$ $\phi$").Double()
		
		h_electron_pT_Trigger_m = hist.Hist.new.Regular(50,0,1000,label = r"e $p_T$ [GeV]").Double()
		h_Leadingelectron_pT_Trigger_m = hist.Hist.new.Regular(50,0,1000,label = r"e $p_T$ [GeV]").Double()
		h_electron_eta_Trigger_m = hist.Hist.new.Regular(20,-4,4,label = r"e $\eta$").Double()
		h_electron_phi_Trigger_m = hist.Hist.new.Regular(20,-pi,pi,label = r"e $\phi$").Double()
		h_muon_pT_Trigger_m = hist.Hist.new.Regular(50,0,1000,label = r"$\mu$ $p_T$ [GeV]").Double()
		h_Leadingmuon_pT_Trigger_m = hist.Hist.new.Regular(50,0,1000,label = r"$\mu$ $p_T$ [GeV]").Double()
		h_muon_eta_Trigger_m = hist.Hist.new.Regular(20,-4,4,label = r"$\mu$ $\eta$").Double()
		h_muon_phi_Trigger_m = hist.Hist.new.Regular(20,-pi,pi,label = r"$\mu$ $\phi$").Double()

		#Basic Kinematic histograms Jets (check which Jets most useful based on 
		h_Jet_pT_Trigger_h = hist.Hist.new.Regular(50,0,1000,label = r"Jet $p_T$ [GeV]").Double()
		h_LeadingJet_pT_Trigger_h = hist.Hist.new.Regular(50,0,1000,label = r"Jet $p_T$ [GeV]").Double()
		h_Jet_eta_Trigger_h = hist.Hist.new.Regular(20,-4,4,label = r"Jet $\eta$").Double()
		h_Jet_phi_Trigger_h = hist.Hist.new.Regular(20,-pi,pi,label = r"Jet $\phi$").Double()
		h_AK8Jet_pT_Trigger_h = hist.Hist.new.Regular(50,0,1000,label = r"AK8Jet $p_T$ [GeV]").Double()
		h_LeadingAK8Jet_pT_Trigger_h = hist.Hist.new.Regular(50,0,1000,label = r"AK8Jet $p_T$ [GeV]").Double()
		h_AK8Jet_eta_Trigger_h = hist.Hist.new.Regular(20,-4,4,label = r"AK8Jet $\eta$").Double()
		h_AK8Jet_phi_Trigger_h = hist.Hist.new.Regular(20,-pi,pi,label = r"AK8Jet $\phi$").Double()
		
		h_Jet_pT_Trigger_e = hist.Hist.new.Regular(50,0,1000,label = r"Jet $p_T$ [GeV]").Double()
		h_LeadingJet_pT_Trigger_e = hist.Hist.new.Regular(50,0,1000,label = r"Jet $p_T$ [GeV]").Double()
		h_Jet_eta_Trigger_e = hist.Hist.new.Regular(20,-4,4,label = r"Jet $\eta$").Double()
		h_Jet_phi_Trigger_e = hist.Hist.new.Regular(20,-pi,pi,label = r"Jet $\phi$").Double()
		h_AK8Jet_pT_Trigger_e = hist.Hist.new.Regular(50,0,1000,label = r"AK8Jet $p_T$ [GeV]").Double()
		h_LeadingAK8Jet_pT_Trigger_e = hist.Hist.new.Regular(50,0,1000,label = r"AK8Jet $p_T$ [GeV]").Double()
		h_AK8Jet_eta_Trigger_e = hist.Hist.new.Regular(20,-4,4,label = r"AK8Jet $\eta$").Double()
		h_AK8Jet_phi_Trigger_e = hist.Hist.new.Regular(20,-pi,pi,label = r"AK8Jet $\phi$").Double()
		
		h_Jet_pT_Trigger_m = hist.Hist.new.Regular(50,0,1000,label = r"Jet $p_T$ [GeV]").Double()
		h_LeadingJet_pT_Trigger_m = hist.Hist.new.Regular(50,0,1000,label = r"Jet $p_T$ [GeV]").Double()
		h_Jet_eta_Trigger_m = hist.Hist.new.Regular(20,-4,4,label = r"Jet $\eta$").Double()
		h_Jet_phi_Trigger_m = hist.Hist.new.Regular(20,-pi,pi,label = r"Jet $\phi$").Double()
		h_AK8Jet_pT_Trigger_m = hist.Hist.new.Regular(50,0,1000,label = r"AK8Jet $p_T$ [GeV]").Double()
		h_LeadingAK8Jet_pT_Trigger_m = hist.Hist.new.Regular(50,0,1000,label = r"AK8Jet $p_T$ [GeV]").Double()
		h_AK8Jet_eta_Trigger_m = hist.Hist.new.Regular(20,-4,4,label = r"AK8Jet $\eta$").Double()
		h_AK8Jet_phi_Trigger_m = hist.Hist.new.Regular(20,-pi,pi,label = r"AK8Jet $\phi$").Double()

		#Add MET histogram
		h_MET_Trigger_h = hist.Hist.new.Regular(10,0,500,label=r"MET [GeV]").Double()
		h_MET_Trigger_e = hist.Hist.new.Regular(10,0,500,label=r"MET [GeV]").Double()
		h_MET_Trigger_m = hist.Hist.new.Regular(10,0,500,label=r"MET [GeV]").Double()


		cutflow_dict = dict.fromkeys(["Sample","PreSkimming","Skimming","Trigger","Tau_pT","Tau_eta","decay","deepboosted","Mass_Cut","Higgs_dR"])
		cutflow_dict["Sample"] = dataset
		cutflow_dict["PreSkimming"] = numEvents_Dict[dataset] 
		cutflow_dict["Skimming"] = ak.num(boostedtau,axis=0)
		
		#Obtain the cross section scale factor	
		if (self.isData):
			CrossSec_Weight = 1 
		else:
			CrossSec_Weight = weight_calc(dataset,sumWEvents_Dict[dataset])
		
		#HLT Triggers
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

		#Trigger Offline selection
		tau = tau[event_level.pfMET > 180]
		boostedtau = boostedtau[event_level.pfMET > 180]
		AK8Jet = AK8Jet[event_level.pfMET > 180]
		Jet = Jet[event_level.pfMET > 180]
		electron = electron[event_level.pfMET > 180]
		muon = muon[event_level.pfMET > 180]
		event_level = event_level[event_level.pfMET > 180]				
			
		#PreSelection Cuts
		#AK8Jets
	#	AK8Jet = AK8Jet[AK8Jet.pt > 200]
	#	AK8Jet = AK8Jet[np.abs(AK8Jet.eta) < 2.5]
	#	AK8Jet = AK8Jet[AK8Jet.softDropM < 30]
	#	AK8Jet = AK8Jet[AK8Jet.Id > 1]

	#	#Require only events with atleast 1 AK8 Jet
	#	tau = tau[ak.num(AK8Jet,axis=1) > 0]
	#	boostedtau = boostedtau[ak.num(AK8Jet,axis=1) > 0]
	#	Jet = Jet[ak.num(AK8Jet,axis=1) > 0]
	#	electron = electron[ak.num(AK8Jet,axis=1) > 0]
	#	muon = muon[ak.num(AK8Jet,axis=1) > 0]
	#	event_level = event_level[ak.num(AK8Jet,axis=1) > 0]				
	#	AK8Jet = AK8Jet[ak.num(AK8Jet,axis=1) > 0]

	#	#Force AK8Jet to be pT ordered
	#	AK8Jet = AK8Jet[ak.argsort(-AK8Jet.pt,axis=1)]

	#	
	#	#Boosted Tau selections
	#	boostedtau = boostedtau[boostedtau.pt > 20]
	#	boostedtau = boostedtau[np.abs(boostedtau.eta) < 2.5]
	#	boostedtau = boostedtau[boostedtau.DBT >= 0.85]
	#	
	#	#Tau
	#	tau = tau[tau.pt > 20]
	#	tau = tau[np.abs(tau.eta) < 2.5]
	#	tau = tau[tau.IDVsJets > 1]
	#	tau = tau[tau.IDVsEle > 1 ]
	#	tau = tau[tau.IDVsMu > 1 ]

	#	#Handle events with differing numbers of taus without loosing anything
	#	tau_1 = tau[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	muon_1 = muon[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	electron_1 = electron[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	AK8Jet_1 = AK8Jet[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	Jet_1 = Jet[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	event_level_1 = event_level[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	boostedtau_1 = boostedtau[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	
	#	tau_2 = tau[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	muon_2 = muon[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	electron_2 = electron[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	AK8Jet_2 = AK8Jet[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	Jet_2 = Jet[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	event_level_2 = event_level[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	boostedtau_2 = boostedtau[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	
	#	tau_3 = tau[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]
	#	muon_3 = muon[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]
	#	electron_3 = electron[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]
	#	AK8Jet_3 = AK8Jet[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]
	#	Jet_3 = Jet[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]
	#	event_level_3 = event_level[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]
	#	boostedtau_3 = boostedtau[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]

	#	#Boosted Tau Cross Cleaning
	#	boostedtau_1 = boostedtau_1[crossClean(boostedtau_1, muon_1, 0.05)]
	#	boostedtau_1 = boostedtau_1[crossClean(boostedtau_1, electron_1, 0.05)]
	#	boostedtau_1 = boostedtau_1[crossClean_PartJet(boostedtau_1, AK8Jet_1[:,0], 1.5)]
	#	
	#	boostedtau_3 = boostedtau_3[crossClean(boostedtau_3, muon_3, 0.05)]
	#	boostedtau_3 = boostedtau_3[crossClean(boostedtau_3, electron_3, 0.05)]
	#	boostedtau_3 = boostedtau_3[crossClean_PartJet(boostedtau_3, AK8Jet_3[:,0], 1.5)]
	#	
	#	#Tau Cross Cleaning
	#	tau_2 = tau_2[crossClean(tau_2, muon_2, 0.05)]
	#	tau_2 = tau_2[crossClean(tau_2, electron_2, 0.05)]
	#	tau_2 = tau_2[crossClean_PartJet(tau_2, AK8Jet_2[:,0], 1.5)]
	#	
	#	tau_3 = tau_3[crossClean(tau_3, muon_3, 0.05)]
	#	tau_3 = tau_3[crossClean(tau_3, electron_3, 0.05)]
	#	tau_3 = tau_3[crossClean_PartJet(tau_3, AK8Jet_3[:,0], 1.5)]

	#	#Recombine everything
	#	tau = ak.concatenate([tau_1,ak.concatenate([tau_2,tau_3])])
	#	muon = ak.concatenate([muon_1,ak.concatenate([muon_2,muon_3])])
	#	electron = ak.concatenate([electron_1,ak.concatenate([electron_2,electron_3])])
	#	AK8Jet = ak.concatenate([AK8Jet_1,ak.concatenate([AK8Jet_2,AK8Jet_3])])
	#	Jet = ak.concatenate([Jet_1,ak.concatenate([Jet_2,Jet_3])])
	#	event_level = ak.concatenate([event_level_1,ak.concatenate([event_level_2,event_level_3])])
	#	boostedtau = ak.concatenate([boostedtau_1,ak.concatenate([boostedtau_2,boostedtau_3])])

	#	#Electrons
	#	electron = electron[electron.pt > 10]
	#	electron = electron[np.abs(electron.eta) < 2.5]
	#	iso_cond1 = np.bitwise_and(electron.RelIso < 0.112*ak.ones_like(electron.pt) + (0.506/electron.pt),np.abs(electron.eta) <= 1.479)		
	#	iso_cond2 = np.bitwise_and(electron.RelIso < 0.108*ak.ones_like(electron.pt) + 0.963/electron.pt,np.bitwise_and(np.abs(electron.eta) > 1.479, np.abs(electron.eta) <= 2.5))
	#	iso_cond_e = np.bitwise_or(iso_cond1,iso_cond2)
	#	electron = electron[iso_cond_e]

	#	electron = electron[np.bitwise_not(np.bitwise_and(np.abs(electron.eta) < 1.57, np.abs(electron.eta) > 1.44))] #Veto ECal tansition region 
	#	
	#	#Electron Cross Cleaning
	#	electron = electron[crossClean_PartJet(electron,AK8Jet[:,0],0.8)]

	#	#Muons
	#	muon = muon[muon.pt > 15]
	#	muon = muon[np.abs(muon.eta) < 2.4]
	#	muon = muon[muon.LooseId]
	#	muon = muon[muon.RelIso < 0.25]
	#	
	#	#Muon Cross Cleaning
	#	muon = muon[crossClean_PartJet(muon,AK8Jet[:,0],0.8)]

	#	#Jets (AK4)
	#	Jet = Jet[Jet.pt > 30]
	#	Jet = Jet[np.abs(Jet.eta) < 2.5]
	#	Jet = Jet[Jet.JetId > 1]

	#	#Jet Cross Cleaning
	#	Jet = Jet[crossClean_DiJet(Jet,AK8Jet[:,0],1.2)]

	#	#Split objects for non tau lepton cross cleaning
	#	tau_1 = tau[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) > 0)]
	#	muon_1 = muon[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) > 0)]
	#	electron_1 = electron[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) > 0)]
	#	AK8Jet_1 = AK8Jet[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) > 0)]
	#	Jet_1 = Jet[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) > 0)]
	#	event_level_1 = event_level[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) > 0)]
	#	boostedtau_1 = boostedtau[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) > 0)]

	#	tau_2 = tau[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) == 0)]
	#	muon_2 = muon[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) == 0)]
	#	electron_2 = electron[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) == 0)]
	#	AK8Jet_2 = AK8Jet[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) == 0)]
	#	Jet_2 = Jet[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) == 0)]
	#	event_level_2 = event_level[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) == 0)]
	#	boostedtau_2 = boostedtau[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) == 0)]

	#	tau_3 = tau[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) > 0)]
	#	muon_3 = muon[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) > 0)]
	#	electron_3 = electron[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) > 0)]
	#	AK8Jet_3 = AK8Jet[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) > 0)]
	#	Jet_3 = Jet[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) > 0)]
	#	event_level_3 = event_level[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) > 0)]
	#	boostedtau_3 = boostedtau[np.bitwise_and(ak.num(muon,axis=1) > 0,ak.num(electron,axis=1) > 0)]

	#	tau_4 = tau[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) == 0)]
	#	muon_4 = muon[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) == 0)]
	#	electron_4 = electron[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) == 0)]
	#	AK8Jet_4 = AK8Jet[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) == 0)]
	#	Jet_4 = Jet[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) == 0)]
	#	event_level_4 = event_level[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) == 0)]
	#	boostedtau_4 = boostedtau[np.bitwise_and(ak.num(muon,axis=1) == 0,ak.num(electron,axis=1) == 0)]

	#	Jet_1 = Jet_1[crossClean_JetPart(Jet_1,electron_1[:,0],0.4)]
	#	Jet_3 = Jet_3[crossClean_JetPart(Jet_3,electron_3[:,0],0.4)]
	#	
	#	Jet_2 = Jet_2[crossClean_JetPart(Jet_2,muon_2[:,0],0.4)]
	#	Jet_3 = Jet_3[crossClean_JetPart(Jet_3,muon_3[:,0],0.4)]

	#	#Recombine everything
	#	tau = ak.concatenate([tau_1,ak.concatenate([tau_2,ak.concatenate([tau_3,tau_4])])])
	#	muon = ak.concatenate([muon_1,ak.concatenate([muon_2,ak.concatenate([muon_3,muon_4])])])
	#	electron = ak.concatenate([electron_1,ak.concatenate([electron_2,ak.concatenate([electron_3,electron_4])])])
	#	AK8Jet = ak.concatenate([AK8Jet_1,ak.concatenate([AK8Jet_2,ak.concatenate([AK8Jet_3,AK8Jet_4])])])
	#	Jet = ak.concatenate([Jet_1,ak.concatenate([Jet_2,ak.concatenate([Jet_3,Jet_4])])])
	#	event_level = ak.concatenate([event_level_1,ak.concatenate([event_level_2,ak.concatenate([event_level_3,event_level_4])])])
	#	boostedtau = ak.concatenate([boostedtau_1,ak.concatenate([boostedtau_2,ak.concatenate([boostedtau_3,boostedtau_4])])])

	#	#Split objects for lepton cross cleaning of taus
	#	tau_1 = tau[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	muon_1 = muon[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	electron_1 = electron[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	AK8Jet_1 = AK8Jet[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	Jet_1 = Jet[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	event_level_1 = event_level[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	boostedtau_1 = boostedtau[np.bitwise_and(ak.num(tau,axis=1) == 0,ak.num(boostedtau,axis=1) > 0)]
	#	
	#	tau_2 = tau[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	muon_2 = muon[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	electron_2 = electron[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	AK8Jet_2 = AK8Jet[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	Jet_2 = Jet[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	event_level_2 = event_level[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	boostedtau_2 = boostedtau[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) == 0)]
	#	
	#	tau_3 = tau[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]
	#	muon_3 = muon[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]
	#	electron_3 = electron[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]
	#	AK8Jet_3 = AK8Jet[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]
	#	Jet_3 = Jet[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]
	#	event_level_3 = event_level[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]
	#	boostedtau_3 = boostedtau[np.bitwise_and(ak.num(tau,axis=1) > 0,ak.num(boostedtau,axis=1) > 0)]

	#	Jet_1 = Jet_1[crossClean_JetPart(Jet_1,boostedtau_1[:,0],0.4)]
	#	Jet_3 = Jet_3[crossClean_JetPart(Jet_3,boostedtau_3[:,0],0.4)]

	#	Jet_2 = Jet_2[crossClean_JetPart(Jet_2,tau_2[:,0],0.4)]
	#	Jet_3 = Jet_3[crossClean_JetPart(Jet_3,tau_3[:,0],0.4)]

	#	#Recombine everything
	#	tau = ak.concatenate([tau_1,ak.concatenate([tau_2,tau_3])])
	#	muon = ak.concatenate([muon_1,ak.concatenate([muon_2,muon_3])])
	#	electron = ak.concatenate([electron_1,ak.concatenate([electron_2,electron_3])])
	#	AK8Jet = ak.concatenate([AK8Jet_1,ak.concatenate([AK8Jet_2,AK8Jet_3])])
	#	Jet = ak.concatenate([Jet_1,ak.concatenate([Jet_2,Jet_3])])
	#	event_level = ak.concatenate([event_level_1,ak.concatenate([event_level_2,event_level_3])])
	#	boostedtau = ak.concatenate([boostedtau_1,ak.concatenate([boostedtau_2,boostedtau_3])])
		

		#Split output channels
		#Fully Hadronic Channel
		#Split into hps taus and boosted taus
		tau_hps = tau[ak.num(tau,axis=1) > 1]
		muon_hps = muon[ak.num(tau,axis=1) > 1]
		electron_hps = electron[ak.num(tau,axis=1) > 1]
		AK8Jet_hps = AK8Jet[ak.num(tau,axis=1) > 1]
		Jet_hps = Jet[ak.num(tau,axis=1) > 1]
		event_level_hps = event_level[ak.num(tau,axis=1) > 1]
		boostedtau_hps = boostedtau[ak.num(tau,axis=1) > 1]
		
		tau_boost = tau[ak.num(boostedtau,axis=1) > 1]
		muon_boost = muon[ak.num(boostedtau,axis=1) > 1]
		electron_boost = electron[ak.num(boostedtau,axis=1) > 1]
		AK8Jet_boost = AK8Jet[ak.num(boostedtau,axis=1) > 1]
		Jet_boost = Jet[ak.num(boostedtau,axis=1) > 1]
		event_level_boost = event_level[ak.num(boostedtau,axis=1) > 1]
		boostedtau_boost = boostedtau[ak.num(boostedtau,axis=1) > 1]

		#HPS taus
		tau_h_dR_cond = deltaR_Selec(tau_hps[:,0],tau_hps[:,1],1.5) 
		tau_h_mass_cond = dimass_Selec(tau_hps[:,0],tau_hps[:,1],20)
		tau_h_cond = np.bitwise_and(tau_h_dR_cond, tau_h_mass_cond)
	
		boostedtau_h_dR_cond = deltaR_Selec(boostedtau_boost[:,0],boostedtau_boost[:,1],1.5)
		boostedtau_h_mass_cond = dimass_Selec(boostedtau_boost[:,0],boostedtau_boost[:,1],20)
		boostedtau_h_cond = np.bitwise_and(boostedtau_h_dR_cond,boosted_tau_h_mass_cond)
		
		tau_hps = tau_hps[tau_h_cond]
		muon_hps = muon_hps[tau_h_cond]
		electron_hps = electron_hps[tau_h_cond]
		AK8Jet_hps = AK8Jet_hps[tau_h_cond]
		Jet_hps = Jet_hps[tau_h_cond]
		event_level_hps = event_level_hps[tau_h_cond]
		boostedtau_hps = boostedtau_hps[tau_h_cond]

		tau_boost = tau_boost[boostedtau_h_cond]
		muon_boost = muon_boost[boostedtau_h_cond]
		electron_boost = electron_boost[boostedtau_h_cond]
		AK8Jet_boost = AK8Jet_boost[boostedtau_h_cond]
		Jet_boost = Jet_boost[boostedtau_h_cond]
		event_level_boost = event_level_boost[boostedtau_h_cond]
		boostedtau_boost = boostedtau_boost[boostedtau_h_cond]
		
		boostedtau_h = ak.concatenate([boostedtau_hps,boostedtau_boost])
		tau_h = ak.concatenate([tau_hps,tau_boost])
		electron_h = ak.concatenate([electron_hps,electron_boost])
		muon_h = ak.concatenate([muon_hps,muon_boost])
		Jet_h = ak.concatenate([Jet_hps,Jet_boost])
		AK8Jet_h = ak.concatenate([AK8Jet_hps,AK8Jet_boost])
		event_level_h = ak.concatenate([event_level_hps,event_level_boost])

		#MET Phi selection
		deltaPhiMET_Cond_h = deltaPhi_METSelec(AK8Jet_h[:,0],event_level,1)
		boostedtau_h = boostedtau[deltaPhiMET_Cond_h]
		tau_h = tau[deltaPhiMET_Cond_h]
		electron_h = electron[deltaPhiMET_Cond_h]
		muon_h = muon[deltaPhiMET_Cond_h]
		Jet_h = Jet[deltaPhiMET_Cond_h]
		AK8Jet_h = AK8Jet[deltaPhiMET_Cond_h]
		event_level_h = event_level[deltaPhiMET_Cond_h]
		
		#Electron Channel
		#Split into only canidate events for e tau channel
		tau_e = tau[ak.num(electron,axis=1) > 0]
		muon_e = muon[ak.num(electron,axis=1) > 0]
		electron_e = electron[ak.num(electron,axis=1) > 0]
		AK8Jet_e = AK8Jet[ak.num(electron,axis=1) > 0]
		Jet_e = Jet[ak.num(electron,axis=1) > 0]
		event_level_e = event_level[ak.num(electron,axis=1) > 0]
		boostedtau_e = boostedtau[ak.num(electron,axis=1) > 0]
		
		#Make slections and apply them
		tau_e_dR_cond = deltaR_Selec(tau_e[:,0],electron_e[:,0],1.5)  
		tau_e_mass_cond = dimass_Selec(tau_e[:,0],electron_e[:,0],20)
		tau_e_cond = np.bitwise_and(tau_e_dR_cond, tau_e_mass_cond)
		
		boostedtau_e_dR_cond = deltaR_Selec(boostedtau_e[:,0],electron_e[:,0],1.5)
		boostedtau_e_mass_cond = dimass_Selec(boostedtau_e[:,0],electron_e[:,0],20)
		boostedtau_e_cond = np.bitwise_and(boostedtau_e_dR_cond,boosted_tau_e_mass_cond)
		
		electron_Cond = np.bitwise_or(tau_e_cond,boostedtau_e_cond)
		boostedtau_e = boostedtau_e[electron_Cond]
		tau_e = tau_e[electron_Cond]
		electron_e = electron_e[electron_Cond]
		muon_e = muon_e[electron_Cond]
		Jet_e = Jet_e[electron_Cond]
		AK8Jet_e = AK8Jet_e[electron_Cond]
		event_level_e = event_level_e[electron_Cond]
		
		#MET Phi selection
		deltaPhiMET_Cond_e = deltaPhi_METSelec(AK8Jet_e[:,0],event_level,1)
		boostedtau_e = boostedtau[deltaPhiMET_Cond_e]
		tau_e = tau[deltaPhiMET_Cond_e]
		electron_e = electron[deltaPhiMET_Cond_e]
		muon_e = muon[deltaPhiMET_Cond_e]
		Jet_e = Jet[deltaPhiMET_Cond_e]
		AK8Jet_e = AK8Jet[deltaPhiMET_Cond_e]
		event_level_e = event_level[deltaPhiMET_Cond_e]
		
		#Muon Channel
		#Split into only canidate events for muon tau channel
		tau_m = tau[ak.num(muon,axis=1) > 0]
		muon_m = muon[ak.num(muon,axis=1) > 0]
		electron_m = electron[ak.num(muon,axis=1) > 0]
		AK8Jet_m = AK8Jet[ak.num(muon,axis=1) > 0]
		Jet_m = Jet[ak.num(muon,axis=1) > 0]
		event_level_m = event_level[ak.num(muon,axis=1) > 0]
		boostedtau_m = boostedtau[ak.num(muon,axis=1) > 0]
		
		tau_m_dR_cond = deltaR_Selec(tau_m[:,0],muon_m[:,0],1.5)  
		tau_m_mass_cond = dimass_Selec(tau_m[:,0],muon_m[:,0],20)
		tau_m_cond = np.bitwise_and(tau_m_dR_cond, tau_m_mass_cond)
		
		boostedtau_m_dR_cond = deltaR_Selec(boostedtau_m[:,0],muon_m[:,0],1.5)
		boostedtau_m_mass_cond = dimass_Selec(boostedtau_m[:,0],muon_m[:,0],20)
		boostedtau_m_cond = np.bitwise_and(boostedtau_m_dR_cond,boosted_tau_m_mass_cond)
		
		muon_Cond = np.bitwise_or(tau_m_cond,boostedtau_m_cond)
		boostedtau_m = boostedtau_m[muon_Cond]
		tau_m = tau_m[muon_Cond]
		electron_m = electron_m[muon_Cond]
		muon_m = muon_m[muon_Cond]
		Jet_m = Jet_m[muon_Cond]
		AK8Jet_m = AK8Jet_m[muon_Cond]
		event_level_m = event_level_m[muon_Cond]
		
		#MET Phi selection
		deltaPhiMET_Cond_m = deltaPhi_METSelec(AK8Jet_m[:,0],event_level,1)
		boostedtau_m = boostedtau[deltaPhiMET_Cond_m]
		tau_m = tau[deltaPhiMET_Cond_m]
		electron_m = electron[deltaPhiMET_Cond_m]
		muon_m = muon[deltaPhiMET_Cond_m]
		Jet_m = Jet[deltaPhiMET_Cond_m]
		AK8Jet_m = AK8Jet[deltaPhiMET_Cond_m]
		event_level_m = event_level[deltaPhiMET_Cond_m]
		
		#Fill histograms after to trigger and all selections
		#Boosted Taus
		h_boostedtau_pT_Trigger_h.fill(ak.ravel(boostedtau_h.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(boostedtau_h.pt))[0]))
		h_Leadingboostedtau_pT_Trigger_h.fill(ak.ravel(boostedtau_h[:,0].pt),weight=ak.ravel(event_level_h.event_weight*CrossSec_Weight))
		h_boostedtau_eta_Trigger_h.fill(ak.ravel(boostedtau_h.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(boostedtau_h.eta))[0]))
		h_boostedtau_phi_Trigger_h.fill(ak.ravel(boostedtau_h.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(boostedtau_h.phi))[0]))
		h_boostedtau_raw_iso_Trigger_h.fill(ak.ravel(boostedtau_h.iso),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(boostedtau_h.iso))[0]))

		
		h_boostedtau_pT_Trigger_e.fill(ak.ravel(boostedtau_e.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(boostedtau_e.pt))[0]))
		h_Leadingboostedtau_pT_Trigger_e.fill(ak.ravel(boostedtau_e[:,0].pt),weight=ak.ravel(event_level_e.event_weight*CrossSec_Weight))
		h_boostedtau_eta_Trigger_e.fill(ak.ravel(boostedtau_e.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(boostedtau_e.eta))[0]))
		h_boostedtau_phi_Trigger_e.fill(ak.ravel(boostedtau_e.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(boostedtau_e.phi))[0]))
		h_boostedtau_raw_iso_Trigger_e.fill(ak.ravel(boostedtau_e.iso),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(boostedtau_e.iso))[0]))
		
		
		h_boostedtau_pT_Trigger_m.fill(ak.ravel(boostedtau_m.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(boostedtau_m.pt))[0]))
		h_Leadingboostedtau_pT_Trigger_m.fill(ak.ravel(boostedtau_m[:,0].pt),weight=ak.ravel(event_level_m.event_weight*CrossSec_Weight))
		h_boostedtau_eta_Trigger_m.fill(ak.ravel(boostedtau_m.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(boostedtau_m.eta))[0]))
		h_boostedtau_phi_Trigger_m.fill(ak.ravel(boostedtau_m.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(boostedtau_m.phi))[0]))
		h_boostedtau_raw_iso_Trigger_m.fill(ak.ravel(boostedtau_m.iso),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(boostedtau_m.iso))[0]))
		

		#HPS Taus
		h_tau_pT_Trigger_h.fill(ak.ravel(tau_h.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(tau_h.pt))[0]))
		h_Leadingtau_pT_Trigger_h.fill(ak.ravel(tau_h[:,0].pt),weight=ak.ravel(event_level_h.event_weight*CrossSec_Weight))
		h_tau_eta_Trigger_h.fill(ak.ravel(tau_h.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(tau_h.eta))[0]))
		h_tau_phi_Trigger_h.fill(ak.ravel(tau_h.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(tau_h.phi))[0]))
		
		h_tau_pT_Trigger_e.fill(ak.ravel(tau_e.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(tau_e.pt))[0]))
		h_Leadingtau_pT_Trigger_e.fill(ak.ravel(tau_e[:,0].pt),weight=ak.ravel(event_level_e.event_weight*CrossSec_Weight))
		h_tau_eta_Trigger_e.fill(ak.ravel(tau_e.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(tau_e.eta))[0]))
		h_tau_phi_Trigger_e.fill(ak.ravel(tau_e.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(tau_e.phi))[0]))
		
		h_tau_pT_Trigger_m.fill(ak.ravel(tau_m.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(tau_m.pt))[0]))
		h_Leadingtau_pT_Trigger_m.fill(ak.ravel(tau_m[:,0].pt),weight=ak.ravel(event_level_m.event_weight*CrossSec_Weight))
		h_tau_eta_Trigger_m.fill(ak.ravel(tau_m.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(tau_m.eta))[0]))
		h_tau_phi_Trigger_m.fill(ak.ravel(tau_m.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(tau_m.phi))[0]))
		
		#Electrons
		h_electron_pT_Trigger_h.fill(ak.ravel(electron_h.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(electron_h.pt))[0]))
		h_Leadingelectron_pT_Trigger_h.fill(ak.ravel(electron_h[:,0].pt),weight=ak.ravel(event_level_h.event_weight*CrossSec_Weight))
		h_electron_eta_Trigger_h.fill(ak.ravel(electron_h.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(electron_h.eta))[0]))
		h_electron_phi_Trigger_h.fill(ak.ravel(electron_h.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(electron_h.phi))[0]))

		h_electron_pT_Trigger_e.fill(ak.ravel(electron_e.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(electron_e.pt))[0]))
		h_Leadingelectron_pT_Trigger_e.fill(ak.ravel(electron_e[:,0].pt),weight=ak.ravel(event_level_e.event_weight*CrossSec_Weight))
		h_electron_eta_Trigger_e.fill(ak.ravel(electron_e.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(electron_e.eta))[0]))
		h_electron_phi_Trigger_e.fill(ak.ravel(electron_e.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(electron_e.phi))[0]))
		
		h_electron_pT_Trigger_m.fill(ak.ravel(electron_m.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(electron_m.pt))[0]))
		h_Leadingelectron_pT_Trigger_m.fill(ak.ravel(electron_m[:,0].pt),weight=ak.ravel(event_level_m.event_weight*CrossSec_Weight))
		h_electron_eta_Trigger_m.fill(ak.ravel(electron_m.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(electron_m.eta))[0]))
		h_electron_phi_Trigger_m.fill(ak.ravel(electron_m.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(electron_m.phi))[0]))
		#Muons
		h_muon_pT_Trigger_h.fill(ak.ravel(muon_h.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(muon_h.pt))[0]))
		h_Leadingmuon_pT_Trigger_h.fill(ak.ravel(muon_h.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(muon_h.pt))[0]))
		h_muon_eta_Trigger_h.fill(ak.ravel(muon_h.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(muon_h.eta))[0]))
		h_muon_phi_Trigger_h.fill(ak.ravel(muon_h.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(muon_h.phi))[0]))

		h_muon_pT_Trigger_e.fill(ak.ravel(muon_e.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(muon_e.pt))[0]))
		h_Leadingmuon_pT_Trigger_e.fill(ak.ravel(muon_e.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(muon_e.pt))[0]))
		h_muon_eta_Trigger_e.fill(ak.ravel(muon_e.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(muon_e.eta))[0]))
		h_muon_phi_Trigger_e.fill(ak.ravel(muon_e.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(muon_e.phi))[0]))

		h_muon_pT_Trigger_m.fill(ak.ravel(muon_m.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(muon_m.pt))[0]))
		h_Leadingmuon_pT_Trigger_m.fill(ak.ravel(muon_m.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(muon_m.pt))[0]))
		h_muon_eta_Trigger_m.fill(ak.ravel(muon_m.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(muon_m.eta))[0]))
		h_muon_phi_Trigger_m.fill(ak.ravel(muon_m.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(muon_m.phi))[0]))


		#Jets 
		h_Jet_pT_Trigger_h.fill(ak.ravel(Jet_h.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(Jet_h.pt))[0]))
		h_LeadingJet_pT_Trigger_h.fill(ak.ravel(Jet_h[:,0].pt),weight=ak.ravel(event_level_h.event_weight*CrossSec_Weight))
		h_Jet_eta_Trigger_h.fill(ak.ravel(Jet_h.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(Jet_h.eta))[0]))
		h_Jet_phi_Trigger_h.fill(ak.ravel(Jet_h.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(Jet_h.phi))[0]))
		
		h_Jet_pT_Trigger_e.fill(ak.ravel(Jet_e.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(Jet_e.pt))[0]))
		h_LeadingJet_pT_Trigger_e.fill(ak.ravel(Jet_e[:,0].pt),weight=ak.ravel(event_level_e.event_weight*CrossSec_Weight))
		h_Jet_eta_Trigger_e.fill(ak.ravel(Jet_e.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(Jet_e.eta))[0]))
		h_Jet_phi_Trigger_e.fill(ak.ravel(Jet_e.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(Jet_e.phi))[0]))
		
		h_Jet_pT_Trigger_m.fill(ak.ravel(Jet_m.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(Jet_m.pt))[0]))
		h_LeadingJet_pT_Trigger_m.fill(ak.ravel(Jet_m[:,0].pt),weight=ak.ravel(event_level_m.event_weight*CrossSec_Weight))
		h_Jet_eta_Trigger_m.fill(ak.ravel(Jet_m.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(Jet_m.eta))[0]))
		h_Jet_phi_Trigger_m.fill(ak.ravel(Jet_m.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(Jet_m.phi))[0]))

		#AK8/Fat Jets
		h_AK8Jet_pT_Trigger_h.fill(ak.ravel(AK8Jet_h.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet_h.pt))[0]))
		h_LeadingAK8Jet_pT_Trigger_h.fill(ak.ravel(AK8Jet_h[:,0].pt),weight=ak.ravel(event_level_h.event_weight*CrossSec_Weight))
		h_AK8Jet_eta_Trigger_h.fill(ak.ravel(AK8Jet_h.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet_h.eta))[0]))
		h_AK8Jet_phi_Trigger_h.fill(ak.ravel(AK8Jet_h.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_h.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet_h.phi))[0]))
		
		h_AK8Jet_pT_Trigger_e.fill(ak.ravel(AK8Jet_e.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet_e.pt))[0]))
		h_LeadingAK8Jet_pT_Trigger_e.fill(ak.ravel(AK8Jet_e[:,0].pt),weight=ak.ravel(event_level_e.event_weight*CrossSec_Weight))
		h_AK8Jet_eta_Trigger_e.fill(ak.ravel(AK8Jet_e.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet_e.eta))[0]))
		h_AK8Jet_phi_Trigger_e.fill(ak.ravel(AK8Jet_e.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_e.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet_e.phi))[0]))
		
		h_AK8Jet_pT_Trigger_m.fill(ak.ravel(AK8Jet_m.pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet_m.pt))[0]))
		h_LeadingAK8Jet_pT_Trigger_m.fill(ak.ravel(AK8Jet_m[:,0].pt),weight=ak.ravel(event_level_m.event_weight*CrossSec_Weight))
		h_AK8Jet_eta_Trigger_m.fill(ak.ravel(AK8Jet_m.eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet_m.eta))[0]))
		h_AK8Jet_phi_Trigger_m.fill(ak.ravel(AK8Jet_m.phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level_m.event_weight*CrossSec_Weight),ak.ones_like(AK8Jet_m.phi))[0]))

		#Print the MET
		h_MET_Trigger_h.fill(ak.ravel(event_level_h.pfMET),weight=ak.ravel(event_level_h.eventweight*CrossSec_Weight))
		h_MET_Trigger_e.fill(ak.ravel(event_level_e.pfMET),weight=ak.ravel(event_level_e.eventweight*CrossSec_Weight))
		h_MET_Trigger_m.fill(ak.ravel(event_level_m.pfMET),weight=ak.ravel(event_level_m.eventweight*CrossSec_Weight))
		
		return{
			dataset: {
				#"Weight": CrossSec_Weight,
				"Weight_Val": CrossSec_Weight,
				"Weight": ak.to_list(event_level.event_weight*CrossSec_Weight), 
				#Boosted Tau kineamtic distirubtions
				"boostedtau_pt_Trigg_h": h_boostedtau_pT_Trigger_h,
				"Leadingboostedtau_pt_Trigg_h": h_Leadingboostedtau_pT_Trigger_h,
				"boostedtau_eta_Trigg_h": h_boostedtau_eta_Trigger_h,
				"boostedtau_phi_Trigg_h": h_boostedtau_phi_Trigger_h,
				"boostedtau_iso_Trigg_h": h_boostedtau_raw_iso_Trigger_h,
				
				"boostedtau_pt_Trigg_e": h_boostedtau_pT_Trigger_e,
				"Leadingboostedtau_pt_Trigg_e": h_Leadingboostedtau_pT_Trigger_e,
				"boostedtau_eta_Trigg_e": h_boostedtau_eta_Trigger_e,
				"boostedtau_phi_Trigg_e": h_boostedtau_phi_Trigger_e,
				"boostedtau_iso_Trigg_e": h_boostedtau_raw_iso_Trigger_e,
				
				"boostedtau_pt_Trigg_m": h_boostedtau_pT_Trigger_m,
				"Leadingboostedtau_pt_Trigg_m": h_Leadingboostedtau_pT_Trigger_m,
				"boostedtau_eta_Trigg_m": h_boostedtau_eta_Trigger_m,
				"boostedtau_phi_Trigg_m": h_boostedtau_phi_Trigger_m,
				"boostedtau_iso_Trigg_m": h_boostedtau_raw_iso_Trigger_m,
				
				#Boosted Tau kineamtic distirubtions
				"boostedtau_pt_Trigg_h": h_boostedtau_pT_Trigger_h,
				"Leadingboostedtau_pt_Trigg_h": h_Leadingboostedtau_pT_Trigger_h,
				"boostedtau_eta_Trigg_h": h_boostedtau_eta_Trigger_h,
				"boostedtau_phi_Trigg_h": h_boostedtau_phi_Trigger_h,
				"boostedtau_iso_Trigg_h": h_boostedtau_raw_iso_Trigger_h,
				
				"boostedtau_pt_Trigg_e": h_boostedtau_pT_Trigger_e,
				"Leadingboostedtau_pt_Trigg_e": h_Leadingboostedtau_pT_Trigger_e,
				"boostedtau_eta_Trigg_e": h_boostedtau_eta_Trigger_e,
				"boostedtau_phi_Trigg_e": h_boostedtau_phi_Trigger_e,
				"boostedtau_iso_Trigg_e": h_boostedtau_raw_iso_Trigger_e,
				
				"boostedtau_pt_Trigg_m": h_boostedtau_pT_Trigger_m,
				"Leadingboostedtau_pt_Trigg_m": h_Leadingboostedtau_pT_Trigger_m,
				"boostedtau_eta_Trigg_m": h_boostedtau_eta_Trigger_m,
				"boostedtau_phi_Trigg_m": h_boostedtau_phi_Trigger_m,
				"boostedtau_iso_Trigg_m": h_boostedtau_raw_iso_Trigger_m,
				
				#Electron kineamtic distirubtions
				"electron_pt_Trigg_h": h_electron_pT_Trigger_h,
				"Leadingelectron_pt_Trigg_h": h_Leadingelectron_pT_Trigger_h,
				"electron_eta_Trigg_h": h_electron_eta_Trigger_h,
				"electron_phi_Trigg_h": h_electron_phi_Trigger_h,
				
				"electron_pt_Trigg_e": h_electron_pT_Trigger_e,
				"Leadingelectron_pt_Trigg_e": h_Leadingelectron_pT_Trigger_e,
				"electron_eta_Trigg_e": h_electron_eta_Trigger_e,
				"electron_phi_Trigg_e": h_electron_phi_Trigger_e,
				
				"electron_pt_Trigg_m": h_electron_pT_Trigger_m,
				"Leadingelectron_pt_Trigg_m": h_Leadingelectron_pT_Trigger_m,
				"electron_eta_Trigg_m": h_electron_eta_Trigger_m,
				"electron_phi_Trigg_m": h_electron_phi_Trigger_m,
				
				#Muon kineamtic distirubtions
				"muon_pt_Trigg_h": h_muon_pT_Trigger_h,
				"Leadingmuon_pt_Trigg_h": h_Leadingmuon_pT_Trigger_h,
				"muon_eta_Trigg_h": h_muon_eta_Trigger_h,
				"muon_phi_Trigg_h": h_muon_phi_Trigger_h,
				
				"muon_pt_Trigg_e": h_muon_pT_Trigger_e,
				"Leadingmuon_pt_Trigg_e": h_Leadingmuon_pT_Trigger_e,
				"muon_eta_Trigg_e": h_muon_eta_Trigger_e,
				"muon_phi_Trigg_e": h_muon_phi_Trigger_e,
				
				"muon_pt_Trigg_m": h_muon_pT_Trigger_m,
				"Leadingmuon_pt_Trigg_m": h_Leadingmuon_pT_Trigger_m,
				"muon_eta_Trigg_m": h_muon_eta_Trigger_m,
				"muon_phi_Trigg_m": h_muon_phi_Trigger_m,
				
				#Jet kineamtic distirubtions
				"Jet_pt_Trigg_h": h_Jet_pT_Trigger_h,
				"LeadingJet_pt_Trigg_h": h_LeadingJet_pT_Trigger_h,
				"Jet_eta_Trigg_h": h_Jet_eta_Trigger_h,
				"Jet_phi_Trigg_h": h_Jet_phi_Trigger_h,
				
				"Jet_pt_Trigg_e": h_Jet_pT_Trigger_e,
				"LeadingJet_pt_Trigg_e": h_LeadingJet_pT_Trigger_e,
				"Jet_eta_Trigg_e": h_Jet_eta_Trigger_e,
				"Jet_phi_Trigg_e": h_Jet_phi_Trigger_e,
				
				"Jet_pt_Trigg_m": h_Jet_pT_Trigger_m,
				"LeadingJet_pt_Trigg_m": h_LeadingJet_pT_Trigger_m,
				"Jet_eta_Trigg_m": h_Jet_eta_Trigger_m,
				"Jet_phi_Trigg_m": h_Jet_phi_Trigger_m,
				
				#AK8Jet kineamtic distirubtions
				"AK8Jet_pt_Trigg_h": h_AK8Jet_pT_Trigger_h,
				"LeadingAK8Jet_pt_Trigg_h": h_LeadingAK8Jet_pT_Trigger_h,
				"AK8Jet_eta_Trigg_h": h_AK8Jet_eta_Trigger_h,
				"AK8Jet_phi_Trigg_h": h_AK8Jet_phi_Trigger_h,
				
				"AK8Jet_pt_Trigg_e": h_AK8Jet_pT_Trigger_e,
				"LeadingAK8Jet_pt_Trigg_e": h_LeadingAK8Jet_pT_Trigger_e,
				"AK8Jet_eta_Trigg_e": h_AK8Jet_eta_Trigger_e,
				"AK8Jet_phi_Trigg_e": h_AK8Jet_phi_Trigger_e,

				"AK8Jet_pt_Trigg_m": h_AK8Jet_pT_Trigger_m,
				"LeadingAK8Jet_pt_Trigg_m": h_LeadingAK8Jet_pT_Trigger_m,
				"AK8Jet_eta_Trigg_m": h_AK8Jet_eta_Trigger_m,
				"AK8Jet_phi_Trigg_m": h_AK8Jet_phi_Trigger_m,

				#Print MET
				"MET_h": h_MET_Trigger_h,
				"MET_e": h_MET_Trigger_e,
				"MET_m": h_MET_Trigger_m,
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
			 memory="7 GB",
			disk="2.5 GB",
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
            #chunksize=150000,
            #maxchunks = 10
		)
	else: #Iterative runner
		runner = processor.Runner(executor = processor.IterativeExecutor(), schema=BaseSchema)
	
	four_tau_hist_list = [
			"tau_pt_Trigg_h","tau_eta_Trigg_h","tau_phi_Trigg_h",
			"electron_pt_Trigg_h","electron_eta_Trigg_h","electron_phi_Trigg_h",
			"muon_pt_Trigg_h","muon_eta_Trigg_h","muon_phi_Trigg_h",
			"Jet_pt_Trigg_h","Jet_eta_Trigg_h","Jet_phi_Trigg_h",
			"AK8Jet_pt_Trigg_h","AK8Jet_eta_Trigg_h","AK8Jet_phi_Trigg_h","MET_h",
			"tau_pt_Trigg_e","tau_eta_Trigg_e","tau_phi_Trigg_e",
			"electron_pt_Trigg_e","electron_eta_Trigg_e","electron_phi_Trigg_e",
			"muon_pt_Trigg_e","muon_eta_Trigg_e","muon_phi_Trigg_e",
			"Jet_pt_Trigg_e","Jet_eta_Trigg_e","Jet_phi_Trigg_e",
			"AK8Jet_pt_Trigg_e","AK8Jet_eta_Trigg_e","AK8Jet_phi_Trigg_e","MET_e",
			"tau_pt_Trigg_m","tau_eta_Trigg_m","tau_phi_Trigg_m",
			"electron_pt_Trigg_m","electron_eta_Trigg_m","electron_phi_Trigg_m",
			"muon_pt_Trigg_m","muon_eta_Trigg_m","muon_phi_Trigg_m",
			"Jet_pt_Trigg_m","Jet_eta_Trigg_m","Jet_phi_Trigg_m",
			"AK8Jet_pt_Trigg_m","AK8Jet_eta_Trigg_m","AK8Jet_phi_Trigg_m","MET_m",
			] 

	hist_name_dict = {
					"tau_pt_Trigg_h": r"$\tau$ $p_T$ after Trigger",
					"tau_eta_Trigg_h": r"$\tau$ $\eta$ after Trigger","tau_phi_Trigg_h": r"$\tau$ $\phi$ after Trigger", 
					"electron_pt_Trigg_h": r"e $p_T$ after Trigger",
					"electron_eta_Trigg_h": r"e $\eta$ after Trigger", "electron_phi_Trigg_h": r"e $\phi$ after Trigger", 
					"muon_pt_Trigg_h": r"$\mu$ $p_T$ after Trigger",
					"muon_eta_Trigg_h": r"$\mu$ $\eta$ after Trigger","muon_phi_Trigg_h": r"$\mu$ $\phi$ after Trigger", 
					"Jet_pt_Trigg_h": r"Jet $p_T$ after Trigger",
					"Jet_eta_Trigg_h": r"Jet $\eta$ after Trigger", "Jet_phi_Trigg_h": r"Jet $\phi$ after Trigger", 
					"AK8Jet_pt_Trigg_h": r"AK8Jet $p_T$ after Trigger",
					"AK8Jet_eta_Trigg_h": r"AK8Jet $\eta$ after Trigger","AK8Jet_phi_Trigg_h": r"AK8Jet $\phi$ after Trigger",
					"MET_h": r"MET after Trigger",
					
					"tau_pt_Trigg_e": r"$\tau$ $p_T$ after Trigger",
					"tau_eta_Trigg_e": r"$\tau$ $\eta$ after Trigger","tau_phi_Trigg_e": r"$\tau$ $\phi$ after Trigger", 
					"electron_pt_Trigg_e": r"e $p_T$ after Trigger",
					"electron_eta_Trigg_e": r"e $\eta$ after Trigger", "electron_phi_Trigg_e": r"e $\phi$ after Trigger", 
					"muon_pt_Trigg_e": r"$\mu$ $p_T$ after Trigger",
					"muon_eta_Trigg_e": r"$\mu$ $\eta$ after Trigger","muon_phi_Trigg_e": r"$\mu$ $\phi$ after Trigger", 
					"Jet_pt_Trigg_e": r"Jet $p_T$ after Trigger",
					"Jet_eta_Trigg_e": r"Jet $\eta$ after Trigger", "Jet_phi_Trigg_e": r"Jet $\phi$ after Trigger", 
					"AK8Jet_pt_Trigg_e": r"AK8Jet $p_T$ after Trigger",
					"AK8Jet_eta_Trigg_e": r"AK8Jet $\eta$ after Trigger","AK8Jet_phi_Trigg_e": r"AK8Jet $\phi$ after Trigger",
					"MET_e": r"MET after Trigger",
					
					"tau_pt_Trigg_m": r"$\tau$ $p_T$ after Trigger",
					"tau_eta_Trigg_m": r"$\tau$ $\eta$ after Trigger","tau_phi_Trigg_m": r"$\tau$ $\phi$ after Trigger", 
					"electron_pt_Trigg_m": r"e $p_T$ after Trigger",
					"electron_eta_Trigg_m": r"e $\eta$ after Trigger", "electron_phi_Trigg_m": r"e $\phi$ after Trigger", 
					"muon_pt_Trigg_m": r"$\mu$ $p_T$ after Trigger",
					"muon_eta_Trigg_m": r"$\mu$ $\eta$ after Trigger","muon_phi_Trigg_m": r"$\mu$ $\phi$ after Trigger", 
					"Jet_pt_Trigg_m": r"Jet $p_T$ after Trigger",
					"Jet_eta_Trigg_m": r"Jet $\eta$ after Trigger", "Jet_phi_Trigg_m": r"Jet $\phi$ after Trigger", 
					"AK8Jet_pt_Trigg_m": r"AK8Jet $p_T$ after Trigger",
					"AK8Jet_eta_Trigg_m": r"AK8Jet $\eta$ after Trigger","AK8Jet_phi_Trigg_m": r"AK8Jet $\phi$ after Trigger",
					"MET_m": r"MET after Trigger",
					}

	#Diretory for files
	Skimmed_Ganesh_base = "root://cmsxrootd.hep.wisc.edu//store/user/gparida/HHbbtt/Hadded_Skimmed_Files/Full_Production_CMSSW_13_0_13_Nov24_23/LooseSelection_MET_gt_80_nFatJet_gt_0_Skim/2018/"
	
	file_dict_test = {
			"ZZ4l": [Skimmed_Ganesh_base + "ZZTo4L.root"],
			"Data_MET": [Skimmed_Ganesh_base + "MET/MET_Run2018A.root"]
		}
	
	file_dict_data = {
			"Data_MET": [Skimmed_Ganesh_base + "MET/MET_Run2018A.root",Skimmed_Ganesh_base + "MET/MET_Run2018B.root",Skimmed_Ganesh_base + "MET/MET_Run2018C.root",
				Skimmed_Ganesh_base + "MET/MET_Run2018D.root",Skimmed_Ganesh_base + "MET/MET_Run2018D_2.root",Skimmed_Ganesh_base + "MET/MET_Run2018D_3.root",
				Skimmed_Ganesh_base + "MET/MET_Run2018D_4.root"]
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
			#QCD Samples
			"QCD_HT50to100": [Skimmed_Ganesh_base + "QCD_HT50to100.root"], "QCD_HT100to200": [Skimmed_Ganesh_base + "QCD_HT100to200.root"], 
			"QCD_HT200to300": [Skimmed_Ganesh_base + "QCD_HT200to300.root"], "QCD_HT300to500": [Skimmed_Ganesh_base + "QCD_HT300to500.root"],
			"QCD_HT500to700": [Skimmed_Ganesh_base + "QCD_HT500to700.root"], "QCD_HT700to1000": [Skimmed_Ganesh_base + "QCD_HT700to1000.root"],
			"QCD_HT1000to1500": [Skimmed_Ganesh_base + "QCD_HT1000to1500.root"], "QCD_HT1500to2000": [Skimmed_Ganesh_base + "QCD_HT1500to2000.root"],
			"QCD_HT2000toInf": [Skimmed_Ganesh_base + "QCD_HT2000toInf.root"],
			"Data_MET": [Skimmed_Ganesh_base + "MET/MET_Run2018A.root",Skimmed_Ganesh_base + "MET/MET_Run2018B.root",Skimmed_Ganesh_base + "MET/MET_Run2018C.root",
				Skimmed_Ganesh_base + "MET/MET_Run2018D.root",Skimmed_Ganesh_base + "MET/MET_Run2018D_2.root",Skimmed_Ganesh_base + "MET/MET_Run2018D_3.root",
				Skimmed_Ganesh_base + "MET/MET_Run2018D_4.root"]
		}
	
	#Background lists 
	background_list_full = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$","QCD"] #Full background list
	background_list_test = [r"$ZZ \rightarrow 4l$"] #Only ZZ4l background for testing
	background_list_none = [] #No backgrounds for data only testing
	
	#Set file dictionary and list of backgrounds prior to running processor
	file_dict = file_dict_full
	background_list = background_list_full

	start_time = time.time()
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
	trigger_name = "MET_Trigger"
	four_tau_names = {
		"tau_pt_Trigg_h": "Tau_pT_Trigger_hadronic" + "-" + trigger_name,
		"tau_eta_Trigg_h": "Tau_eta_Trigger_hadronic" + "-" + trigger_name,
		"tau_phi_Trigg_h": "Tau_phi_Trigger_hadronic" + "-" + trigger_name,
		"tau_iso_Trigg_h": "Tau_iso_Trigger_hadronic" + "-" + trigger_name,
		"electron_pt_Trigg_h": "Electron_pT_Trigger_hadronic" + "-" + trigger_name,
		"electron_eta_Trigg_h": "Electron_eta_Trigger_hadronic" + "-" + trigger_name,
		"electron_phi_Trigg_h": "Electron_phi_Trigger_hadronic" + "-" + trigger_name,
		"muon_pt_Trigg_h": "Muon_pT_Trigger_hadronic" + "-" + trigger_name,
		"muon_eta_Trigg_h": "Muon_eta_Trigger_hadronic" + "-" + trigger_name,
		"muon_phi_Trigg_h": "Muon_phi_Trigger_hadronic" + "-" + trigger_name,
		"Jet_pt_Trigg_h": "Jet_pT_Trigger_hadronic" + "-" + trigger_name,
		"Jet_eta_Trigg_h": "Jet_eta_Trigger_hadronic" + "-" + trigger_name,
		"Jet_phi_Trigg_h": "Jet_phi_Trigger_hadronic" + "-" + trigger_name,
		"AK8Jet_pt_Trigg_h": "AK8Jet_pT_Trigger_hadronic" + "-" + trigger_name,
		"AK8Jet_eta_Trigg_h": "AK8Jet_eta_Trigger_hadronic" + "-" + trigger_name,
		"AK8Jet_phi_Trigg_h": "AK8Jet_phi_Trigger_hadronic" + "-" + trigger_name,
		"MET_h": "MET_Trigger_hadronic" + "-" + trigger_name,
		
		"tau_pt_Trigg_e": "Tau_pT_Trigger_electron" + "-" + trigger_name,
		"tau_eta_Trigg_e": "Tau_eta_Trigger_electron" + "-" + trigger_name,
		"tau_phi_Trigg_e": "Tau_phi_Trigger_electron" + "-" + trigger_name,
		"tau_iso_Trigg_e": "Tau_iso_Trigger_electron" + "-" + trigger_name,
		"electron_pt_Trigg_e": "Electron_pT_Trigger_electron" + "-" + trigger_name,
		"electron_eta_Trigg_e": "Electron_eta_Trigger_electron" + "-" + trigger_name,
		"electron_phi_Trigg_e": "Electron_phi_Trigger_electron" + "-" + trigger_name,
		"muon_pt_Trigg_e": "Muon_pT_Trigger_electron" + "-" + trigger_name,
		"muon_eta_Trigg_e": "Muon_eta_Trigger_electron" + "-" + trigger_name,
		"muon_phi_Trigg_e": "Muon_phi_Trigger_electron" + "-" + trigger_name,
		"Jet_pt_Trigg_e": "Jet_pT_Trigger_electron" + "-" + trigger_name,
		"Jet_eta_Trigg_e": "Jet_eta_Trigger_electron" + "-" + trigger_name,
		"Jet_phi_Trigg_e": "Jet_phi_Trigger_electron" + "-" + trigger_name,
		"AK8Jet_pt_Trigg_e": "AK8Jet_pT_Trigger_electron" + "-" + trigger_name,
		"AK8Jet_eta_Trigg_e": "AK8Jet_eta_Trigger_electron" + "-" + trigger_name,
		"AK8Jet_phi_Trigg_e": "AK8Jet_phi_Trigger_electron" + "-" + trigger_name,
		"MET_e": "MET_Trigger_electron" + "-" + trigger_name,
		
		"tau_pt_Trigg_m": "Tau_pT_Trigger_muon" + "-" + trigger_name,
		"tau_eta_Trigg_m": "Tau_eta_Trigger_muon" + "-" + trigger_name,
		"tau_phi_Trigg_m": "Tau_phi_Trigger_muon" + "-" + trigger_name,
		"tau_iso_Trigg_m": "Tau_iso_Trigger_muon" + "-" + trigger_name,
		"electron_pt_Trigg_m": "Electron_pT_Trigger_muon" + "-" + trigger_name,
		"electron_eta_Trigg_m": "Electron_eta_Trigger_muon" + "-" + trigger_name,
		"electron_phi_Trigg_m": "Electron_phi_Trigger_muon" + "-" + trigger_name,
		"muon_pt_Trigg_m": "Muon_pT_Trigger_muon" + "-" + trigger_name,
		"muon_eta_Trigg_m": "Muon_eta_Trigger_muon" + "-" + trigger_name,
		"muon_phi_Trigg_m": "Muon_phi_Trigger_muon" + "-" + trigger_name,
		"Jet_pt_Trigg_m": "Jet_pT_Trigger_muon" + "-" + trigger_name,
		"Jet_eta_Trigg_m": "Jet_eta_Trigger_muon" + "-" + trigger_name,
		"Jet_phi_Trigg_m": "Jet_phi_Trigger_muon" + "-" + trigger_name,
		"AK8Jet_pt_Trigg_m": "AK8Jet_pT_Trigger_muon" + "-" + trigger_name,
		"AK8Jet_eta_Trigg_m": "AK8Jet_eta_Trigger_muon" + "-" + trigger_name,
		"AK8Jet_phi_Trigg_m": "AK8Jet_phi_Trigger_muon" + "-" + trigger_name,
		"MET_m": "MET_Trigger_muon" + "-" + trigger_name,
	}

	fourtau_out = runner(file_dict, treename="Events", processor_instance=PlottingScriptProcessor()) #Modified for NanoAOD (changd treename)
	end_time = time.time()

	time_running = end_time-start_time
	print("It takes about %.1f s to run"%time_running)

	#Dictionaries of histograms for background, signal and data
	hist_dict_background = dict.fromkeys(four_tau_hist_list)
	hist_dict_signal = dict.fromkeys(four_tau_hist_list)
	hist_dict_data = dict.fromkeys(four_tau_hist_list)
	
	#Save coffea file
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
						#	print("Background: " + background)
						#	print("Sum of entries: %f"%fourtau_out[background][hist_name].sum())
						else:
							crnt_hist += fourtau_out[background][hist_name]
						#	print("Background: " + background)
						#	print("Sum of entries: %f"%fourtau_out[background][hist_name].sum())
						if (background == backgrounds[-1]):
							fig2, ax2 = plt.subplots()
							temp_hist_dict[background_type] = crnt_hist #Try to fix stacking bug
							crnt_hist.plot1d(ax=ax2)
							#if (hist_name == "FourTau_Mass_Arr"):
						#	print("Background: " + background_type)
						#	print("Sum of entries: %f"%crnt_hist.sum())
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
				
		for background in background_list:
			background_array.append(background_stack[background]) 
			print("Background: " + background)
			print("Sum of stacked histogram: %f"%background_stack[background].sum())
					
		#Stack background distributions and plot signal + data distribution
		fig,ax = plt.subplots()
		hep.histplot(background_array,ax=ax,stack=True,histtype="fill",label=background_list,facecolor=TABLEAU_COLORS[:len(background_list)],edgecolor=TABLEAU_COLORS[:len(background_list)])
		#hep.histplot(signal_array,ax=ax,stack=True,histtype="step",label=signal_list,edgecolor=TABLEAU_COLORS[len(background_list)+1],linewidth=2.95)
		hep.histplot(data_array,ax=ax,stack=False,histtype="errorbar", yerr=True,label=["Data"],marker="o",color = "k") #,facecolor='black',edgecolor='black') #,mec='k')
		hep.cms.text("Preliminary",loc=0,fontsize=13)
		#ax.set_title(hist_name_dict[hist_name],loc = "right")
		ax.set_title("2018 Data",loc = "right")
		ax.legend(fontsize=10, loc='upper right')
		#ax.legend(fontsize=10, loc=(1.04,1))
		plt.savefig(four_tau_names[hist_name])
		plt.close()

