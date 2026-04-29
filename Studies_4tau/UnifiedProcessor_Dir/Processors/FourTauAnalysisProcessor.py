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

def weight_calc(sample,numEvents=1):
	return Lumi_2018*xSection_Dictionary[sample]/numEvents

#Use numba to speed up the loop
@numba.njit
def find_Z_Candidates(event_leptons, builder):
	"""
	For a given collection of leptons (with at least 4 leptons) construct and count Z bosons
	"""
	for lepton in event_leptons:
		paired_set = {-1} #Set of already paired indicies
		ZMult = 0
		builder.begin_list()
		for i in range(len(lepton)): #Loop over all leptons
			if i in paired_set: #Avoid double pairings
				continue
			for j in range(i +1, len(lepton)):
				if j in paired_set: #Avoid double pairings
					continue
				if (lepton[i].charge + lepton[j].charge != 0): #Impose 0 eletric charge
					continue
				candidate_mass = np.sqrt((lepton[i].E + lepton[j].E)**2 - (lepton[i].Px + lepton[j].Px)**2 - (lepton[i].Py + lepton[j].Py)**2 - (lepton[i].Pz + lepton[j].Pz)**2)
				if (candidate_mass > 80 and candidate_mass < 100): #Valid Z Boson
					ZMult += 1
					paired_set.add(j) #Add current index to paired set
					break
		
		#Add Z_multiplicity for event to output
		builder.integer(ZMult)
		builder.end_list()
	return builder

def delta_phi(vec1,vec2):
	return (vec1.phi - vec2.phi + pi) % (2*pi) - pi	

def MET_delta_phi(part1,MET_obj):
	return (part1.phi - MET_obj.MET_Phi + pi) % (2*pi) - pi

def deltaR(part1, part2):
	return np.sqrt((part2.eta - part1.eta)**2 + (delta_phi(part1,part2))**2)

def single_mass(part1):
	return np.sqrt((part1.E)**2 - (part1.Px)**2 - (part1.Py)**2 - (part1.Pz)**2)

def four_mass(part_arr): #Four Particle mass assuming each event has 4 particles
	return np.sqrt((part_arr[0].E + part_arr[1].E + part_arr[2].E + part_arr[3].E)**2 - (part_arr[0].Px + part_arr[1].Px + part_arr[2].Px + part_arr[3].Px)**2 - 
		(part_arr[0].Py + part_arr[1].Py + part_arr[2].Py + part_arr[3].Py)**2 - 
		(part_arr[0].Pz + part_arr[1].Pz + part_arr[2].Pz + part_arr[3].Pz)**2)

class Analysis4TauProcessor(processor.ProcessorABC):
	def __init__(self, sumWEvents_Dict, nBoostedTaus = 0, ApplyTrigger = True): #Additional arguements can be added later
		self.isData = False #Default assumption is MC
		self.nBoostedTau_Selec = nBoostedTaus #Number of tau selections
		self.ApplyTrigger = ApplyTrigger
		#self.numEvents_Dict = numEvents_Dict
		self.sumWEvents_Dict = sumWEvents_Dict
		#pass

	def process(self, events):
		vector.register_awkward()
		#Begin by checking if running on data or sample
		dataset = events.metadata['dataset']
		if ("Data_" in dataset): #Check to see if running on data
			print("Is Data")
			self.isData = True	
	
		#Objects for analysis
		event_level = ak.zip(
			{
				"METHTMHT_Trigger": events.HLT_PFHT500_PFMET100_PFMHT100_IDTight,
				"Mu_Trigger": events.HLT_Mu50,
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
				"iso": events.boostedTau_idDeepTau2018v2p7VSjet,
				"DBT": events.boostedTau_rawDeepTau2018v2p7VSjet,
				"decay": events.boostedTau_idDecayModeOldDMs,
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
				"SCEta": events.Electron_deltaEtaSC,
				"IDMVANoIso": events.Electron_mvaNoIso,
				"RelIso": events.Electron_pfRelIso03_all,
					
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
				"JetId": events.Jet_jetId, #Not sure that this is correct
				"eta": events.Jet_eta,
				"phi": events.Jet_phi,
				"mass": events.Jet_mass,
				"nJet": events.nJet,
				#"DeepCSVTags_b": events.Jet_btagCSVV2,
				"DeepCSVTags_b": events.Jet_btagDeepB,
			},
			with_name="PFJetArray",
			behavior=candidate.behavior,
		)
		
		print("!!!=====Dataset=====!!!!")	
		print(type(dataset))
		print(dataset)

		if not(self.isData):
			print("Is MC events are equal to gen weights")
			event_level["event_weight"] = events.genWeight #Set the event weight to the gen weight

		#Control Regions Axis
		#region_array = ["All","ZCR","TCR","FakeCR"]
		region_array = ["All","TCR","NotTCR"]
		#region_array = ["All"] 
		category_axis = hist.axis.StrCategory(region_array, growth=False, name = "category")
		
		#Basic Kinematic histograms Boosted tau
		h_boostedtau_pT_Trigger = hist.Hist.new.Regular(20,0,600,label = r"Boosted $\tau$ $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_Leadingboostedtau_pT_Trigger = hist.Hist.new.Regular(20,0,600,label = r"Boosted $\tau$ Leading $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_Subleadingboostedtau_pT_Trigger = hist.Hist.new.Regular(20,0,600,label = r"Boosted $\tau$ Subleading $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_Thirdleadingboostedtau_pT_Trigger = hist.Hist.new.Regular(20,0,600,label = r"Boosted $\tau$ 3rd-leading $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_Fourthleadingboostedtau_pT_Trigger = hist.Hist.new.Regular(20,0,600,label = r"Boosted $\tau$ 4th-leading $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_boostedtau_eta_Trigger = hist.Hist.new.Regular(20,-4,4,label = r"Boosted $\tau$ $\eta$",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_boostedtau_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi,label = r"Boosted$\tau$ $\phi$",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_boostedtau_raw_iso_Trigger = hist.Hist.new.Regular(20,-1,1,label=r"Raw MVA Score",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		
		#Basic Kinematic histograms leptons (muons and electrons)
		h_electron_pT_Trigger = hist.Hist.new.Regular(15,0,300, label = r"e $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_Leadingelectron_pT_Trigger = hist.Hist.new.Regular(15,0,300, label = r"e Leading $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_electron_eta_Trigger = hist.Hist.new.Regular(20,-4,4, label = r"e $\eta$",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_electron_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi, label = r"e Leading $\phi$",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_muon_pT_Trigger = hist.Hist.new.Regular(15,0,600, label = r"$\mu$ $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_Leadingmuon_pT_Trigger = hist.Hist.new.Regular(15,0,600, label = r"$\mu$ Leading $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_muon_eta_Trigger = hist.Hist.new.Regular(20,-4,4, label = r"$\mu$ $\eta$",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_Leadingmuon_eta_Trigger = hist.Hist.new.Regular(20,-4,4, label = r"$\mu$ $\eta$",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_muon_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi, label = r"$\mu$ $\phi$",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		
		#Basic Kinematic histograms Jets (check which Jets most useful based on 
		h_Jet_pT_Trigger = hist.Hist.new.Regular(50,0,700, label = r"Jet $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_LeadingJet_pT_Trigger = hist.Hist.new.Regular(50,0,700, label = r"Jet Leading $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_Jet_eta_Trigger = hist.Hist.new.Regular(20,-4,4, label = r"Jet $\eta$",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_Jet_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi, label = r"Jet $\phi$",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_AK8Jet_pT_Trigger = hist.Hist.new.Regular(50,0,700, label = r"AK8Jet $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_LeadingAK8Jet_pT_Trigger = hist.Hist.new.Regular(50,0,700, label = r"AK8Jet Leading $p_T$ [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_AK8Jet_eta_Trigger = hist.Hist.new.Regular(20,-4,4, label = r"AK8Jet $\eta$",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_AK8Jet_phi_Trigger = hist.Hist.new.Regular(20,-pi,pi, label = r"AK8Jet $\phi$",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_nAK8Jet_Trigger = hist.Hist.new.Regular(10,0,10, label=r"Number of AK8Jets",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		
		#Add MET, HT and MHT histogram
		h_MET_Trigger = hist.Hist.new.Regular(20,0,500, label=r"MET [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_HT_Trigger = hist.Hist.new.Regular(40,0,1200, label=r"HT [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_MHT_Trigger = hist.Hist.new.Regular(20,0,500, label=r"MHT [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()

		#Add Z Multiplicity and BJet multiplicity
		h_ZMult = hist.Hist.new.Regular(6,0,6, label=r"Z Boson Multiplicity",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_bJetMult = hist.Hist.new.Regular(6,0,6, label = r"b-Jet Multiplicity",overflow = True).StrCat(region_array, growth=False, name = "region").Double()


		#di-boosted tau delta Rs
		h_leading_boostedtau_deltaR = hist.Hist.new.Regular(10,0,5, label = r"Leading boosted \tau pair \Delta R",overflow = True).StrCat(region_array, growth=False, name = "region").Double()
		h_nextleading_boostedtau_deltaR = hist.Hist.new.Regular(10,0,5, label = r"Next leading boosted \tau pair \Delta R",overflow = True).StrCat(region_array, growth=False, name = "region").Double()

		#Reconstructed Mass
		h_FourTau_Mass = hist.Hist.new.Regular(15,0,3000,label=r"Reconstructed Radion Mass [GeV]",overflow = True).StrCat(region_array, growth=False, name = "region").Double()

		#Add cutflow and N-1 tables
		if (self.ApplyTrigger):
			h_CutFlow = hist.Hist.new.StrCategory(["SkimOnly","METCut","nFatJetReq","FlagReq","PVSelec","LeadingBoostedTau","SubleadingBoostedTau","3rdLeadingBoostedTau","4thLeadingBoostedTau","Trigger","VisMassSelec","Higgs_dR"]).Double()
			h_NMinus1 = hist.Hist.new.StrCategory(["SkimOnly","METCut","nFatJetReq","FlagReq","PVSelec","LeadingBoostedTau","SubleadingBoostedTau","3rdLeadingBoostedTau","4thLeadingBoostedTau","Trigger","VisMassSelec","Higgs_dR"]).Double()
		else:
			h_CutFlow = hist.Hist.new.StrCategory(["SkimOnly","METCut","nFatJetReq","FlagReq","PVSelec","LeadingBoostedTau","SubleadingBoostedTau","3rdLeadingBoostedTau","4thLeadingBoostedTau","VisMassSelec","Higgs_dR"]).Double()
			h_NMinus1 = hist.Hist.new.StrCategory(["SkimOnly","METCut","nFatJetReq","FlagReq","PVSelec","LeadingBoostedTau","SubleadingBoostedTau","3rdLeadingBoostedTau","4thLeadingBoostedTau","VisMassSelec","Higgs_dR"]).Double()

		#Fill initial entries in skim and N-1 histograms
		n_Skim = np.size(event_level.nFatJet)
		h_CutFlow.fill("SkimOnly",weight=n_Skim)
		h_NMinus1.fill("SkimOnly",weight=0)
		
		#Obtain the cross section scale factor	
		if (self.isData):
			CrossSec_Weight = 1 
		else:
			CrossSec_Weight = weight_calc(dataset,self.sumWEvents_Dict[dataset])

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
		#n_PreTrigger = n_Skim #Set number of events left before trigger seleciton to PV selection	
		#Temp values of the Tau selections
		n_LeadBoostedTau = -1
		n_SubLeadBoostedTau = -1
		n_3rdLeadBoostedTau = -1
		n_4thLeadBoostedTau = -1

        #Boosted tau selections
		if (self.nBoostedTau_Selec > 0):
			#Impose selections boosted taus
			pT_Cond = boostedtau.pt > 20
			eta_Cond = np.abs(boostedtau.eta) < 2.3
			decayMode_Cond = boostedtau.decay >= 0.5
			#DBT_Iso_Cond = boostedtau.DBT >= 0.9 #0.85
			DBT_Iso_Cond = boostedtau.DBT >= 0.5 #0.85
			
			boostedtau_selec_cond = pT_Cond & eta_Cond & decayMode_Cond & DBT_Iso_Cond
			boostedtau = boostedtau[boostedtau_selec_cond] #Apply selections to all individual taus
		
			#Require events have at least 1 boosted tau
			lead_boostedtau_cond = ak.num(boostedtau,axis=1) >= 1
				
			boostedtau = boostedtau[lead_boostedtau_cond]
			AK8Jet = AK8Jet[lead_boostedtau_cond]
			Jet = Jet[lead_boostedtau_cond]
			electron = electron[lead_boostedtau_cond]
			muon = muon[lead_boostedtau_cond]
			event_level = event_level[lead_boostedtau_cond]

			#Fill post leading tau selection entries in skim and N-1 histograms
			n_LeadBoostedTau = np.size(event_level.nFatJet)
			h_CutFlow.fill("LeadingBoostedTau",weight=n_LeadBoostedTau)
			h_NMinus1.fill("LeadingBoostedTau",weight=n_PreTrigger - n_LeadBoostedTau)

			n_PreTrigger = n_LeadBoostedTau				
			
			#Impose selections on Subleading boosted tau
			if (self.nBoostedTau_Selec > 1):
				#Require events have at least 2 boosted tau
				sublead_boostedtau_cond = ak.num(boostedtau,axis=1) >= 2
					
				boostedtau = boostedtau[sublead_boostedtau_cond]
				AK8Jet = AK8Jet[sublead_boostedtau_cond]
				Jet = Jet[sublead_boostedtau_cond]
				electron = electron[sublead_boostedtau_cond]
				muon = muon[sublead_boostedtau_cond]
				event_level = event_level[sublead_boostedtau_cond]

				#Fill post subl-leading tau selection entries in skim and N-1 histograms
				n_SubLeadBoostedTau = np.size(event_level.nFatJet)
				h_CutFlow.fill("SubleadingBoostedTau",weight=n_SubLeadBoostedTau)
				h_NMinus1.fill("SubleadingBoostedTau",weight=n_LeadBoostedTau - n_SubLeadBoostedTau)

				n_PreTrigger = n_SubLeadBoostedTau				
			
			#Impose selections on third-leading boosted tau
			if (self.nBoostedTau_Selec > 2):
				#Require events have at least 2 boosted tau
				thirdlead_boostedtau_cond = ak.num(boostedtau,axis=1) >= 3
					
				boostedtau = boostedtau[thirdlead_boostedtau_cond]
				AK8Jet = AK8Jet[thirdlead_boostedtau_cond]
				Jet = Jet[thirdlead_boostedtau_cond]
				electron = electron[thirdlead_boostedtau_cond]
				muon = muon[thirdlead_boostedtau_cond]
				event_level = event_level[thirdlead_boostedtau_cond]

				#Fill post 3rd leading tau selection entries in skim and N-1 histograms
				n_3rdLeadBoostedTau = np.size(event_level.nFatJet)
				h_CutFlow.fill("3rdLeadingBoostedTau",weight=n_3rdLeadBoostedTau)
				h_NMinus1.fill("3rdLeadingBoostedTau",weight=n_SubLeadBoostedTau - n_3rdLeadBoostedTau)

				n_PreTrigger = n_3rdLeadBoostedTau				
			
			#Impose selections on fourth-leading boosted tau
			if (self.nBoostedTau_Selec > 3):
				#Require events have at least 2 boosted tau
				fourthlead_boostedtau_cond = ak.num(boostedtau,axis=1) >= 4
					
				boostedtau = boostedtau[fourthlead_boostedtau_cond]
				AK8Jet = AK8Jet[fourthlead_boostedtau_cond]
				Jet = Jet[fourthlead_boostedtau_cond]
				electron = electron[fourthlead_boostedtau_cond]
				muon = muon[fourthlead_boostedtau_cond]
				event_level = event_level[fourthlead_boostedtau_cond]

				#Fill post 4th leading tau selection entries in skim and N-1 histograms
				n_4thLeadBoostedTau = np.size(event_level.nFatJet)
				h_CutFlow.fill("4thLeadingBoostedTau",weight=n_4thLeadBoostedTau)
				h_NMinus1.fill("4thLeadingBoostedTau",weight=n_3rdLeadBoostedTau - n_4thLeadBoostedTau)

				n_PreTrigger = n_4thLeadBoostedTau				
		
		#############
		#Trigger and Offline Cuts
		#############
		n_Trigger = 0
		if (self.ApplyTrigger):
			if (self.isData):
				if ("Data_Mu" == dataset):
					#HLT Trigger(s)
					trigger_cond = event_level.Mu_Trigger
		
					boostedtau = boostedtau[trigger_cond]
					AK8Jet = AK8Jet[trigger_cond]
					Jet = Jet[trigger_cond]
					electron = electron[trigger_cond]
					muon = muon[trigger_cond]
					event_level = event_level[trigger_cond]

					#Muon Trigger offline selection
					nMuon_Cond = ak.any(muon.nMu > 0, axis = 1) 
					boostedtau = boostedtau[nMuon_Cond]
					AK8Jet = AK8Jet[nMuon_Cond]
					Jet = Jet[nMuon_Cond]
					electron = electron[nMuon_Cond]
					muon = muon[nMuon_Cond]
					event_level = event_level[nMuon_Cond]				

					pt_cond = ak.any(muon.pt > 52, axis=1)
					boostedtau = boostedtau[pt_cond]
					AK8Jet = AK8Jet[pt_cond]
					Jet = Jet[pt_cond]
					electron = electron[pt_cond]
					muon = muon[pt_cond]
					event_level = event_level[pt_cond]	

					#Apply ID selections on muons
					id_selec = muon[:,0].IDSelec
					
					boostedtau = boostedtau[id_selec]
					AK8Jet = AK8Jet[id_selec]
					Jet = Jet[id_selec]
					electron = electron[id_selec]
					muon = muon[id_selec]
					event_level = event_level[id_selec]	
					
					#Require events NOT to pass JetHT trigger
				#	boostedtau = boostedtau[np.bitwise_not(event_level.METHTMHT_Trigger)]
				#	AK8Jet = AK8Jet[np.bitwise_not(event_level.METHTMHT_Trigger)]
				#	Jet = Jet[np.bitwise_not(event_level.METHTMHT_Trigger)]
				#	electron = electron[np.bitwise_not(event_level.METHTMHT_Trigger)]
				#	muon = muon[np.bitwise_not(event_level.METHTMHT_Trigger)]
				#	event_level = event_level[np.bitwise_not(event_level.METHTMHT_Trigger)]

			#	if ("Data_JetHT" == dataset):
			#		#HLT Trigger(s)
			#		boostedtau = boostedtau[event_level.METHTMHT_Trigger]
			#		AK8Jet = AK8Jet[event_level.METHTMHT_Trigger]
			#		Jet = Jet[event_level.METHTMHT_Trigger]
			#		electron = electron[event_level.METHTMHT_Trigger]
			#		muon = muon[event_level.METHTMHT_Trigger]
			#		event_level = event_level[event_level.METHTMHT_Trigger]

			#		#Offline Cuts
			#		HT_Cond = event_level.HT > 550
			#		MET_Cond = event_level.MET_pt > 110
			#		MHT_Cond = event_level.MHT > 110
			#		HTMETMHT_Selec = HT_Cond & MET_cond & MHT_Cond
			#		
			#		boostedtau = boostedtau[HTMETMHT_Selec]
			#		AK8Jet = AK8Jet[HTMETMHT_Selec]
			#		Jet = Jet[HTMETMHT_Selec]
			#		electron = electron[HTMETMHT_Selec]
			#		muon = muon[HTMETMHT_Selec]
			#		event_level = event_level[HTMETMHT_Selec]
			else:
				#HLT Trigger(s)
				trigger_cond_mu = event_level.Mu_Trigger
		
				boostedtau = boostedtau[trigger_cond_mu]
				AK8Jet = AK8Jet[trigger_cond_mu]
				Jet = Jet[trigger_cond_mu]
				electron = electron[trigger_cond_mu]
				muon = muon[trigger_cond_mu]
				event_level = event_level[trigger_cond_mu]

				#Muon Trigger offline selection
				nMuon_Cond = ak.any(muon.nMu > 0, axis = 1) 
				boostedtau = boostedtau[nMuon_Cond]
				AK8Jet = AK8Jet[nMuon_Cond]
				Jet = Jet[nMuon_Cond]
				electron = electron[nMuon_Cond]
				muon = muon[nMuon_Cond]
				event_level = event_level[nMuon_Cond]				

				pt_cond = ak.any(muon.pt > 52, axis=1)
				boostedtau = boostedtau[pt_cond]
				AK8Jet = AK8Jet[pt_cond]
				Jet = Jet[pt_cond]
				electron = electron[pt_cond]
				muon = muon[pt_cond]
				event_level = event_level[pt_cond]	

				#Apply ID selections on muons
				id_selec = muon[:,0].IDSelec
					
				boostedtau = boostedtau[id_selec]
				AK8Jet = AK8Jet[id_selec]
				Jet = Jet[id_selec]
				electron = electron[id_selec]
				muon = muon[id_selec]
				event_level = event_level[id_selec]	

				#Fail Single Muon trigger and pass Jet HT Trigger
		#		boostedtau_HT = boostedtau[np.bitwise_not(event_level.Mu_Trigger) & event_level.HTMETMHT_Trigger]
		#		AK8Jet_HT = AK8Jet[np.bitwise_not(event_level.Mu_Trigger) & event_level.HTMETMHT_Trigger]
		#		Jet_HT = Jet[np.bitwise_not(event_level.Mu_Trigger) & event_level.HTMETMHT_Trigger]
		#		electron_HT = electron[np.bitwise_not(event_level.Mu_Trigger) & event_level.HTMETMHT_Trigger]
		#		muon_HT = muon[np.bitwise_not(event_level.Mu_Trigger) & event_level.HTMETMHT_Trigger]
		#		event_level_HT = event_level[np.bitwise_not(event_level.Mu_Trigger) & event_level.HTMETMHT_Trigger]
		#			
		#		#Offline Cuts
		#		HT_Cond = event_level_HT.HT > 550
		#		MET_Cond = event_level_HT.MET_pt > 110
		#		MHT_Cond = event_level_HT.MHT > 110
		#		HTMETMHT_Selec = HT_Cond & MET_cond & MHT_Cond
		#		
		#		boostedtau_HT = boostedtau_HT[HTMETMHT_Selec]
		#		AK8Jet_HT = AK8Jet_HT[HTMETMHT_Selec]
		#		Jet_HT = Jet_HT[HTMETMHT_Selec]
		#		electron_HT = electron_HT[HTMETMHT_Selec]
		#		muon_HT = muon_HT[HTMETMHT_Selec]
		#		event_level_HT = event_level_HT[HTMETMHT_Selec]

		#		#Memory management
		#		boostedtau = ak.concatenate((boostedtau_Mu,boostedtau_HT))
		#		AK8Jet = ak.concatenate((AK8Jet_Mu,AK8Jet_HT))
		#		Jet = ak.concatenate((Jet_Mu,Jet_HT))
		#		electron = ak.concatenate((electron_Mu,electron_HT))
		#		muon = ak.concatenate((muon_Mu,muon_HT))
		#		event_level = ak.concatenate((event_level_Mu,event_level_HT))

		#		del boostedtau_Mu, boostedtau_HT
		#		del AK8Jet_Mu, AK8Jet_HT
		#		del Jet_Mu, Jet_HT
		#		del electron_Mu, electron_HT
		#		del muon_Mu, muon_HT
		#		del event_level_Mu, event_level_HT

			#Fill post trigger entries in skim and N-1 histograms
			n_Trigger = np.size(event_level.nFatJet)
			h_CutFlow.fill("Trigger",weight=n_Trigger)
			h_NMinus1.fill("Trigger",weight=n_PreTrigger - n_Trigger)

		#Force muon selection to check if that is the cause of the imbalance
		if (not(self.ApplyTrigger)):
			print("!!No Trigger + Selections!!")
			print("Forcing Muon selections on all objects")
			Muon_Events = ak.num(muon.pt,axis=1) > 0
			boostedtau = boostedtau[Muon_Events]
			AK8Jet = AK8Jet[Muon_Events]
			Jet = Jet[Muon_Events]
			electron = electron[Muon_Events]
			muon = muon[Muon_Events]
			event_level = event_level[Muon_Events]

		
		#############
		#Find 2 valid tau pairings
		#############
#		if (ak.num(event_level,axis=0) > 0): #Iff any events left
#			btau_4vec = ak.zip({"t": boostedtau.E, "x": boostedtau.Px, "y": boostedtau.Py, "z": boostedtau.Pz},with_name="Momentum4D")
#			lead_btau, btau_others = ak.unzip(ak.cartesian([btau_4vec[:,0], btau_4vec], axis = 1, nested = False))
#			deltaR_Arr = ak.values_astype(lead_btau,np.float64).deltaR(ak.values_astype(btau_4vec,np.float64))
#
#			#Remove leading tau from consideration
#			lead_btau = boostedtau[deltaR_Arr == 0]
#			lead_pair_btau = boostedtau[deltaR_Arr != 0]
#			deltaR_Arr = deltaR_Arr[deltaR_Arr != 0]
#
#			#Select tau that minimizes delta R	
#			lead_pair_btau = lead_pair_btau[deltaR_Arr == ak.min(deltaR_Arr,axis=1)] 
#
#			#Remove any events with no paired taus (NOT Convinced you need this!!)
#			boostedtau = boostedtau[ak.num(lead_pair_btau) > 0]
#			btau_4vec = btau_4vec[ak.num(lead_pair_btau) > 0]
#			Jet = Jet[ak.num(lead_pair_btau) > 0]
#			AK8Jet = AK8Jet[ak.num(lead_pair_btau) > 0]
#			muon = muon[ak.num(lead_pair_btau) > 0]
#			electron = electron[ak.num(lead_pair_btau) > 0]
#			event_level = event_level[ak.num(lead_pair_btau) > 0]
#			lead_pair_btau = lead_pair_btau[ak.num(lead_pair_btau) > 0]
#
#			#Store leading boosted tau and paired boosted tau
#			lead_btau_4vec = ak.firsts(ak.zip({"t": lead_btau.E, "x": lead_btau.Px, "y": lead_btau.Py, "z": lead_btau.Pz},with_name="Momentum4D"))
#			lead_pair_btau_4vec = ak.firsts(ak.zip({"t": lead_pair_btau.E, "x": lead_pair_btau.Px, "y": lead_pair_btau.Py, "z": lead_pair_btau.Pz},with_name="Momentum4D"))
#
#			#Drop first pair of boosted taus form consideration
#			rem_btau_req = (ak.values_astype(lead_btau_4vec, np.float64).deltaR(ak.values_astype(btau_4vec, np.float64)) != 0) & (ak.values_astype(lead_pair_btau_4vec, np.float64).deltaR(ak.values_astype(btau_4vec, np.float64)) != 0)
#			btau_rem = boostedtau[rem_btau_req]
#			
#			#Find Second pair
#			rem_btau_4vec = ak.zip({"t": btau_rem.E, "x": btau_rem.Px, "y": btau_rem.Py, "z": btau_rem.Pz},with_name="Momentum4D")
#			next_lead_btau, rem_btau_others = ak.unzip(ak.cartesian([rem_btau_4vec[:,0], rem_btau_4vec], axis = 1, nested = False))
#			deltaR_Arr = ak.values_astype(next_lead_btau,np.float64).deltaR(ak.values_astype(rem_btau_4vec, np.float64))
#
#			next_lead_btau = btau_rem[deltaR_Arr == 0]
#			next_lead_pair_btau = btau_rem[deltaR_Arr != 0]
#			deltaR_Arr = deltaR_Arr[deltaR_Arr != 0]	
#			next_lead_pair_btau = next_lead_pair_btau[deltaR_Arr == ak.min(deltaR_Arr,axis=1)] 
#
#			#Keep only the 2 identified pairs of taus
#			boostedtau = ak.concatenate((lead_btau, lead_pair_btau),axis=1)
#			boostedtau = ak.concatenate((boostedtau, next_lead_btau),axis=1)
#			boostedtau = ak.concatenate((boostedtau, next_lead_pair_btau),axis=1)
#
#			#Memory management
#			del deltaR_Arr, rem_btau_req
#			del btau_4vec, lead_btau_4vec, lead_pair_btau_4vec, rem_btau_4vec
#			del btau_rem, lead_btau, lead_pair_btau, next_lead_btau, next_lead_pair_btau

		#############
		#Selections fromp paired objects
		#############
#		n_VisMass = 0
#		n_DeltaR = 0
#		if (ak.num(event_level,axis=1) > 0): #Only do this if there are any events left
#			#Leading and next leading pair 4-vectors
#			leading_higgs = ak.zip({
#					"x": boostedtau[:,0].Px + boostedtau[:,1].Px,
#					"y": boostedtau[:,0].Py + boostedtau[:,1].Py,
#					"z": boostedtau[:,0].Pz + boostedtau[:,1].Pz,
#					"t": boostedtau[:,0].E + boostedtau[:,1].E
#				},with_name="Momentum4D"
#			)
#				
#			nextleading_higgs = ak.zip({
#					"x": boostedtau[:,2].Px + boostedtau[:,3].Px,
#					"y": boostedtau[:,2].Py + boostedtau[:,3].Py,
#					"z": boostedtau[:,2].Pz + boostedtau[:,3].Pz,
#					"t": boostedtau[:,2].E + boostedtau[:,3].E
#				},with_name="Momentum4D"
#			)
#
#			#Visable mass selection
#			vis_mass1 = leading_higgs.mass
#			vis_mass2 = nextleading_higgs.mass
#			vis_mass_cond = (vis_mass1 >= 10) & (vis_mass2 >= 10)
#
#			boostedtau = boostedtau[vis_mass_cond]
#			AK8Jet = AK8Jet[vis_mass_cond]
#			Jet = Jet[vis_mass_cond]
#			electron = electron[vis_mass_cond]
#			muon = muon[vis_mass_cond]
#			event_level = event_level[vis_mass_cond]
#
#			#Fill post visable mass entries in skim and N-1 histograms
#			n_VisMass = np.size(event_level.nFatJet)
#			h_CutFlow.fill("VisMassSelec",weight=n_VisMass)
#			h_NMinus1.fill("VisMassSelec",weight=n_Trigger - n_VisMass)
#		
#
#		if (ak.num(event_level,axis=1) > 0): #Only do this if there are any events left
#			#Topology selection
#			topo_cond = leading_higgs.deltaR(nextleading_higgs) >= 2
#
#			boostedtau = boostedtau[topo_cond]
#			AK8Jet = AK8Jet[topo_cond]
#			Jet = Jet[topo_cond]
#			electron = electron[topo_cond]
#			muon = muon[topo_cond]
#			event_level = event_level[topo_cond]
#
#			#Fill post visable mass entries in skim and N-1 histograms
#			n_DeltaR = np.size(event_level.nFatJet)
#			h_CutFlow.fill("Higgs_dR",weight=n_DeltaR)
#			h_NMinus1.fill("Higgs_dR",weight=n_VisMass - n_DeltaR)
#
#
#
		#############
		#Z-Multiplicity and b-Jet Multiplicity
		#############
		if (ak.num(event_level,axis=0) > 0): #If there are events left
			print("nBjets")
	#		#Z Multiplicity function
	#		def Z_Mult_Function(lepton,lep_flavor): 
	#			#Make Good muon selection
	#			if (lep_flavor == "mu"):
	#				id_cond = muon.IDSelec 
	#				d0_cond = np.abs(lepton.D0) < 0.045
	#				dz_cond = np.abs(lepton.Dz) < 0.2
	#				good_lepton_cond = id_cond & d0_cond & dz_cond
	#				good_lepton = lepton[good_lepton_cond]
	#				#else:
	#					#good_lepton = lepton
	#			#Make good electron selection
	#			if (lep_flavor == "ele"):
	#				cond1 = (np.abs(lepton.SCEta) <= 0.8) & (lepton.IDMVANoIso > 0.837)
	#				cond2 = ((np.abs(lepton.SCEta) > 0.8) & (np.abs(lepton.SCEta) <= 1.5)) & (electron.IDMVANoIso > 0.715)
	#				cond3 = (np.abs(lepton.SCEta) >= 1.5) & (electron.IDMVANoIso > 0.357)
	#				good_lepton_cond = cond1 | cond2 | cond3
	#				good_lepton = lepton[good_lepton_cond]
	#			
	#			#print("Number of lepton filled events before Z-multiplicty building: %d"%ak.num(good_lepton,axis=0))
	#			Z_Mult = find_Z_Candidates(good_lepton,ak.ArrayBuilder()).snapshot() #Need to add this function to get the mulitplicity function working

	#			return Z_Mult

			#Apply BJet multiplicity selection
			#Apply pt, eta, loose ID, and deep csv tag cut
			#Jet_B = Jet[Jet.JetId > 0.5]
			#Jet_B = Jet_B[Jet_B.pt > 30]
			#Jet_B = Jet_B[np.abs(Jet_B.eta) < 2.4]
			#Jet_B = Jet[Jet.DeepCSVTags_b > 0.7527] #Old selectuion using btagCSVV2
			#Jet_B = Jet[Jet.DeepCSVTags_b > 0.4168] 
			Jet_B = Jet[Jet.DeepCSVTags_b > 0.7665] 
			NumBJets = ak.num(Jet_B,axis=1)
			event_level["nBJets"] = NumBJets

	#		#Get Z_multiplicity	
	#		electron_ZMult = Z_Mult_Function(electron,"ele")
	#		muon_ZMult = Z_Mult_Function(muon,"mu")
	#		event_level["ZMult"] = muon_ZMult + electron_ZMult
	#		event_level["ZMult_e"] = electron_ZMult
	#		event_level["ZMult_mu"] = muon_ZMult

		else:
			print("No BJets")
			event_level["nBJets"] = 0

#		#############
#		#Get boosted tau pair objects and 4 boosted tau object
#		#############
#		if (ak.num(event_level,axis=0) > 0):
#			#Get pair delta R and delta phi Distributions
#			leading_dR_Arr = ak.ravel(deltaR(boostedtau[:,0],boostedtau[:,1]))
#			leading_dPhi_Arr = ak.ravel(delta_phi(boostedtau[:,0],boostedtau[:,1]))
#			nextleading_dR_Arr = ak.ravel(deltaR(boostedtau[:,2],boostedtau[:,3]))
#			nextleading_dPhi_Arr = ak.ravel(delta_phi(boostedtau[:,2],boostedtau[:,3]))
#			
#			#Get the leading Higgs 4-momenta
#			PxLeading = boostedtau[:,0].Px + boostedtau[:,1].Px
#			PyLeading = boostedtau[:,0].Py + boostedtau[:,1].Py
#			PzLeading = boostedtau[:,0].Pz + boostedtau[:,1].Pz
#			ELeading = boostedtau[:,0].E + boostedtau[:,1].E
#			
#			#Get the subleading Higgs 4-momenta
#			PxSubLeading = boostedtau[:,2].Px + boostedtau[:,3].Px
#			PySubLeading = boostedtau[:,2].Py + boostedtau[:,3].Py
#			PzSubLeading = boostedtau[:,2].Pz + boostedtau[:,3].Pz
#			ESubLeading = boostedtau[:,2].E + boostedtau[:,3].E
#
#			#Reconstructed Higgs Objects
#			Higgs_Leading = ak.zip(
#				{
#					"Px" : ak.from_iter(PxLeading),
#					"Py" : ak.from_iter(PyLeading),
#					"Pz" : ak.from_iter(PzLeading),
#					"E" : ak.from_iter(ELeading)
#				}
#			)
#			Higgs_Leading["phi"] = ak.from_iter(np.arctan2(Higgs_Leading.Py,Higgs_Leading.Px))
#			Higgs_Leading["eta"] = ak.from_iter(np.arcsinh(Higgs_Leading.Pz)/np.sqrt(Higgs_Leading.Px**2 + Higgs_Leading.Py**2 + Higgs_Leading.Pz**2))
#			Higgs_NextLeading = ak.zip(
#				{
#					"Px" : ak.from_iter(PxSubLeading),
#					"Py" : ak.from_iter(PySubLeading),
#					"Pz" : ak.from_iter(PzSubLeading),
#					"E" : ak.from_iter(ESubLeading)
#				}
#			)
#			Higgs_NextLeading["phi"] = ak.from_iter(np.arctan2(Higgs_NextLeading.Py,Higgs_NextLeading.Px))
#			Higgs_NextLeading["eta"] = ak.from_iter(np.arcsinh(Higgs_NextLeading.Pz)/np.sqrt(Higgs_NextLeading.Px**2 + Higgs_NextLeading.Py**2 + Higgs_NextLeading.Pz**2))
#
#
#			#Reconstructed Radion
#			Radion_Reco = ak.zip(
#					{
#						"Px": Higgs_Leading.Px + Higgs_NextLeading.Px,
#						"Py": Higgs_Leading.Py + Higgs_NextLeading.Py,
#						"Pz": Higgs_Leading.Pz + Higgs_NextLeading.Pz,
#						"E": Higgs_Leading.E + Higgs_NextLeading.E,
#					}
#			)
#			#Radion_4Vec = vector.LorentzVectov(ak.zip({"t": Radion_Reco.E,"x": Radion_Reco.Px,"y": Radion_Reco.Py,"z": Radion_Reco.Pz},with_name="LorentzVector"))
#			Radion_4Vec = ak.zip({"t": Radion_Reco.E,"x": Radion_Reco.Px,"y": Radion_Reco.Py,"z": Radion_Reco.Pz},with_name="Momentum4D")
#			Radion_Reco["phi"] = ak.from_iter(np.arctan2(Radion_Reco.Py,Radion_Reco.Px))
#			Radion_Reco["eta"] = Radion_4Vec.eta #ak.from_iter(np.arcsinh(Radion_Reco.Pz)/np.sqrt(Radion_Reco.Px**2 + Radion_Reco.Py**2 + Radion_Reco.Pz**2))
#			event_level["Radion_Charge"] = boostedtau[:,0].charge + boostedtau[:,1].charge + boostedtau[:,2].charge + boostedtau[:,3].charge
#			event_level["LeadingPair_Charge"] = boostedtau[:,0].charge + boostedtau[:,1].charge
#			event_level["SubleadingPair_Charge"] = boostedtau[:,2].charge + boostedtau[:,3].charge
#
#			if (len(Higgs_Leading.eta) != 0):
#				#print("Mass Reconstructed")
#				diHiggs_dR_Arr = ak.ravel(deltaR(Higgs_Leading,Higgs_NextLeading))
#				LeadingHiggs_mass_Arr = ak.ravel(single_mass(Higgs_Leading))	
#				SubLeadingHiggs_mass_Arr = ak.ravel(single_mass(Higgs_NextLeading))
#			
#				#Obtain delta R between each Higgs and the radion
#				leadingHiggs_Rad_dR = ak.ravel(deltaR(Higgs_Leading,Radion_Reco))
#				subleadingHiggs_Rad_dR = ak.ravel(deltaR(Higgs_NextLeading,Radion_Reco))
#			
#				#Obtain Delta phi between MET and Each Higgs
#				leadingHiggs_MET_dPhi_Arr = ak.ravel(MET_delta_phi(Higgs_Leading,event_level))
#				subleadingHiggs_MET_dPhi_Arr = ak.ravel(MET_delta_phi(Higgs_NextLeading,event_level))
#			else:
#				#print("Mass Not Reconstructed")
#				diHiggs_dR_Arr = np.array([])
#				LeadingHiggs_mass_Arr = np.array([])
#				SubLeadingHiggs_mass_Arr = np.array([])
#				leadingHiggs_Rad_dR = np.array([])
#				subleadingHiggs_Rad_dR = np.array([])
#				leadingHiggs_MET_dPhi_Arr = np.array([])
#				subleadingHiggs_MET_dPhi_Arr = np.array([])
#
#			#Fill Higgs Delta Phi
#			phi_leading = np.arctan2(PyLeading,PxLeading)
#			phi_subleading = np.arctan2(PySubLeading,PxSubLeading)
#			Higgs_DeltaPhi_Arr = ak.ravel((phi_leading - phi_subleading + np.pi) % (2 * np.pi) - np.pi)
#			radionPT_HiggsReco = np.sqrt((PxLeading + PxSubLeading)**2 + (PyLeading + PySubLeading)**2)
#			radionPT_Arr = ak.ravel(radionPT_HiggsReco)
#
#			#Obtain delat Phi between MET and Radion
#			radionMET_dPhi = ak.ravel(MET_delta_phi(Radion_Reco,event_level))
#
#			FourTau_Mass_Arr = four_mass([boostedtau[:,0],boostedtau[:,1],boostedtau[:,2],boostedtau[:,3]])

		
		#############
		#Fill histograms
		#############
		#if (ak.num(event_level,axis=0) > 0):
		for region in region_array:
			if (region == "ZCR"):
				region_cond = (event_level.ZMult >= 1) & (event_level.nBJets < 1)
			elif (region == "TCR"):
				#print("Top Region")
				#region_cond = (event_level.ZMult) < 1 & (event_level.nBJets >= 1)
				region_cond = event_level.nBJets >= 1
				#print("Sum of true entries: " + str(ak.sum(region_cond)))
			elif (region == "NotTCR"):
				#print("Not Top Region")
				region_cond = event_level.nBJets < 1
				#print("Sum of true entries: " + str(ak.sum(region_cond)))
			elif (region == "FakeCR"):
				region_cond = ((event_level.ZMult < 1) & (event_level.nBJets < 1)) & ((event_level.LeadingPair_Charge != 0) | (event_level.SubleadingPair_Charge != 0))
			else:
				#print("No Control Region")
				#print(region)
				region_cond = ak.ones_like(event_level.event_num) == 1
				#print("Sum of true entries: " + str(ak.sum(region_cond)))
		
			#Boosted Taus
			h_boostedtau_pT_Trigger.fill(ak.ravel(boostedtau[ak.ravel(region_cond)].pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(boostedtau[ak.ravel(region_cond)].pt))[0]), region = region)
			
			if (self.nBoostedTau_Selec >= 1):
				h_Leadingboostedtau_pT_Trigger.fill(ak.ravel(boostedtau[ak.ravel(region_cond)][:,0].pt),weight=ak.ravel(event_level[ak.ravel(region_cond)][ak.num(boostedtau[ak.ravel(region_cond)],axis=1) >= self.nBoostedTau_Selec].event_weight*CrossSec_Weight), region = region)

			if (self.nBoostedTau_Selec >= 2):
				h_Subleadingboostedtau_pT_Trigger.fill(ak.ravel(boostedtau[ak.ravel(region_cond)][:,1].pt),weight=ak.ravel(event_level[ak.ravel(region_cond)][ak.num(boostedtau[ak.ravel(region_cond)],axis=1) >= self.nBoostedTau_Selec].event_weight*CrossSec_Weight), region = region)
			
			if (self.nBoostedTau_Selec >= 3):
				h_Thirdleadingboostedtau_pT_Trigger.fill(ak.ravel(boostedtau[ak.ravel(region_cond)][:,2].pt),weight=ak.ravel(event_level[ak.ravel(region_cond)][ak.num(boostedtau[ak.ravel(region_cond)],axis=1) >= self.nBoostedTau_Selec].event_weight*CrossSec_Weight), region = region)
			
			if (self.nBoostedTau_Selec >= 4):
				h_Fourthleadingboostedtau_pT_Trigger.fill(ak.ravel(boostedtau[ak.ravel(region_cond)][:,3].pt),weight=ak.ravel(event_level[ak.ravel(region_cond)][ak.num(boostedtau[ak.ravel(region_cond)],axis=1) >= self.nBoostedTau_Selec].event_weight*CrossSec_Weight), region = region)
			
			h_boostedtau_eta_Trigger.fill(ak.ravel(boostedtau[ak.ravel(region_cond)].eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(boostedtau[ak.ravel(region_cond)].eta))[0]), region = region)
			h_boostedtau_phi_Trigger.fill(ak.ravel(boostedtau[ak.ravel(region_cond)].phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(boostedtau[ak.ravel(region_cond)].phi))[0]), region = region)
			h_boostedtau_raw_iso_Trigger.fill(ak.ravel(boostedtau[ak.ravel(region_cond)].iso),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(boostedtau[ak.ravel(region_cond)].iso))[0]), region = region)

			#Electrons
			h_electron_pT_Trigger.fill(ak.ravel(electron[ak.ravel(region_cond)].pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(electron[ak.ravel(region_cond)].pt))[0]), region = region)
			#h_Leadingelectron_pT_Trigger.fill(ak.ravel(ak.firsts(electron[ak.ravel(region_cond)].pt)),weight=ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight), region = region)
			h_electron_eta_Trigger.fill(ak.ravel(electron[ak.ravel(region_cond)].eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(electron[ak.ravel(region_cond)].eta))[0]), region = region)
			h_electron_phi_Trigger.fill(ak.ravel(electron[ak.ravel(region_cond)].phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(electron[ak.ravel(region_cond)].phi))[0]), region = region)
			
			#Muons
			h_muon_pT_Trigger.fill(ak.ravel(muon[ak.ravel(region_cond)].pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(muon[ak.ravel(region_cond)].pt))[0]), region = region)
			h_Leadingmuon_pT_Trigger.fill(ak.ravel(ak.firsts(muon[ak.ravel(region_cond)].pt)),weight=ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight), region = region)
			h_muon_eta_Trigger.fill(ak.ravel(muon[ak.ravel(region_cond)].eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(muon[ak.ravel(region_cond)].eta))[0]), region = region)
			h_Leadingmuon_eta_Trigger.fill(ak.ravel(ak.firsts(muon[region_cond].eta)),weight=ak.ravel(event_level[region_cond].event_weight*CrossSec_Weight), region = region)
			h_muon_phi_Trigger.fill(ak.ravel(muon[ak.ravel(region_cond)].phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(muon[ak.ravel(region_cond)].phi))[0]), region = region)

			#Jets 
			h_Jet_pT_Trigger.fill(ak.ravel(Jet[ak.ravel(region_cond)].pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(Jet[ak.ravel(region_cond)].pt))[0]), region = region)
			#h_LeadingJet_pT_Trigger.fill(ak.ravel(Jet[ak.ravel(region_cond)][ak.num(Jet[ak.ravel(region_cond)],axis=1) > 0][:,0].pt),weight=ak.ravel(event_level[ak.ravel(region_cond)][ak.num(Jet[ak.ravel(region_cond)],axis=1) > 0].event_weight[ak.ravel(region_cond)]*CrossSec_Weight), region = region)
			h_Jet_eta_Trigger.fill(ak.ravel(Jet[ak.ravel(region_cond)].eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(Jet[ak.ravel(region_cond)].eta))[0]), region = region)
			h_Jet_phi_Trigger.fill(ak.ravel(Jet[ak.ravel(region_cond)].phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(Jet[ak.ravel(region_cond)].phi))[0]), region = region)
			
			#AK8/Fat Jets
			h_AK8Jet_pT_Trigger.fill(ak.ravel(AK8Jet[ak.ravel(region_cond)].pt),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(AK8Jet[ak.ravel(region_cond)].pt))[0]), region = region)
			#h_LeadingAK8Jet_pT_Trigger.fill(ak.ravel(ak.firsts(AK8Jet[ak.ravel(region_cond)].pt)),weight=ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight), region = region)
			h_AK8Jet_eta_Trigger.fill(ak.ravel(AK8Jet[ak.ravel(region_cond)].eta),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(AK8Jet[ak.ravel(region_cond)].eta))[0]), region = region)
			h_AK8Jet_phi_Trigger.fill(ak.ravel(AK8Jet[ak.ravel(region_cond)].phi),weight=ak.ravel(ak.broadcast_arrays(ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight),ak.ones_like(AK8Jet[ak.ravel(region_cond)].phi))[0]), region = region)
			h_nAK8Jet_Trigger.fill(ak.ravel(event_level[ak.ravel(region_cond)].nFatJet),weight=ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight), region = region)

			#Store MET, HT and MHT
			h_MET_Trigger.fill(ak.ravel(event_level[ak.ravel(region_cond)].MET_pt),weight=ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight), region = region)
			h_HT_Trigger.fill(ak.ravel(event_level[ak.ravel(region_cond)].HT),weight=ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight), region = region)
			h_MHT_Trigger.fill(ak.ravel(event_level[ak.ravel(region_cond)].MHT),weight=ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight), region = region)

			#Store Z and BJet Mupltiplcity
		#		h_ZMult.fill(ak.ravel(event_level[ak.ravel(region_cond)].ZMult),weight=ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight), region = region)
		#		h_bJetMult.fill(ak.ravel(event_level[ak.ravel(region_cond)].nBJets),weight=ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight), region = region)

			#Store Di-boosted tau delta R
		#	h_leading_boostedtau_deltaR.fill(leading_dR_Arr[ak.ravel(region_cond)], weight=ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight), region = region)
		#	h_nextleading_boostedtau_deltaR.fill(nextleading_dR_Arr[ak.ravel(region_cond)], weight=ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight), region = region)

			#Four Mass
		#	h_FourTau_Mass.fill(FourTau_Mass_Arr[ak.ravel(region_cond)], weight=ak.ravel(event_level[ak.ravel(region_cond)].event_weight*CrossSec_Weight), region = region)
		
		return{
			dataset: {
				#"Weight": CrossSec_Weight,
				"Weight_Val": CrossSec_Weight,
				"Weight": ak.to_list(event_level.event_weight*CrossSec_Weight), 
				"Event_Count": np.sum(ak.to_list(event_level.event_weight*CrossSec_Weight)),
				"n_Skim": n_Skim,
				"n_LeadBoostedTau": n_LeadBoostedTau,
				"n_SubLeadBoostedTau": n_SubLeadBoostedTau,
				"n_3rdLeadBoostedTau": n_3rdLeadBoostedTau,
				"n_4thLeadBoostedTau": n_4thLeadBoostedTau,
				"n_Trigger": n_Trigger,
				#"n_VisMass": n_VisMass,
				#"n_Higgs_dR": n_DeltaR,
				
				#Boosted Tau kineamtic distirubtions
				"boostedtau_pt_Trigg": h_boostedtau_pT_Trigger,
				"Leadingboostedtau_pt_Trigg": h_Leadingboostedtau_pT_Trigger,
				"Subleadingboostedtau_pt_Trigg": h_Subleadingboostedtau_pT_Trigger,
				"Thirdleadingboostedtau_pt_Trigg": h_Thirdleadingboostedtau_pT_Trigger,
				"Fourthleadingboostedtau_pt_Trigg": h_Fourthleadingboostedtau_pT_Trigger,
				"boostedtau_eta_Trigg": h_boostedtau_eta_Trigger,
				"boostedtau_phi_Trigg": h_boostedtau_phi_Trigger,
				"boostedtau_iso_Trigg": h_boostedtau_raw_iso_Trigger,
				
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
				#"LeadingJet_pt_Trigg": h_LeadingJet_pT_Trigger,
				"Jet_eta_Trigg": h_Jet_eta_Trigger,
				"Jet_phi_Trigg": h_Jet_phi_Trigger,
				
				#AK8Jet kineamtic distirubtions
				"AK8Jet_pt_Trigg": h_AK8Jet_pT_Trigger,
				"LeadingAK8Jet_pt_Trigg": h_LeadingAK8Jet_pT_Trigger,
				"AK8Jet_eta_Trigg": h_AK8Jet_eta_Trigger,
				"AK8Jet_phi_Trigg": h_AK8Jet_phi_Trigger,
				"nAK8Jet_Trigg": h_nAK8Jet_Trigger, 
				
				#Print MET
				"MET": h_MET_Trigger,
				"HT": h_HT_Trigger,
				"MHT": h_MHT_Trigger,

				#Store the Mini Cutflow and N-1 Table 
				"Mini_Cutflow": h_CutFlow,
				"Mini_NMinus1": h_NMinus1,

				#Multiplicites
			#	"ZMult": h_ZMult,
				"bJetMult": h_bJetMult,

				#Pair Delta Rs
			#	"LeadingPair_dR": h_leading_boostedtau_deltaR,
			#	"NexLeadingPair_dR": h_nextleading_boostedtau_deltaR 
			}
		}

	def postprocess(self, accumulator):
		pass
