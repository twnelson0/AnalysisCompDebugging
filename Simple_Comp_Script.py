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
from cutflow_producer import cutflow_producer
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
						"TTTo2L2Nu": 97.5595, "TTToSemiLeptonic": 381.0923, "TTToHadronic": 365.2482,
						
						#DiBoson Background
						#"ZZ2l2q": 3.22, "WZ3l1nu": 4.708, "WZ2l2q": 5.595, "WZ1l1nu2q": 10.71, "VV2l2nu": 11.95, "WZ1l3nu": 3.05, #"WZ3l1nu.root" : 27.57,
						"ZZ2l2q": 3.676, "WZ3l1nu": , "WZ2l2q": 6.565, "WZ1l1nu2q": 9.119, "WZ1l3nu": 3.414, "VV2l2nu": 11.09, #"WZ3l1nu.root" : 27.57,
						
						#ZZ->4l
						"ZZ4l": 1.325,
						#DiBoson continued
						#"ZZTo2L2Nu_powheg": 0.564, "ZZTo2L2Q_amcNLO": 3.22, "ZZTo4L_powheg": 1.212, "WWTo2L2Nu_powheg": 12.178, "WWTo4Q_powheg": 51.723, "WWTo1LNuQQ_powheg": 49.997, 
						#"WZTo1L3Nu_amcatnloFXFX": 3.033, "WZTo2L2Q_amcNLO": 5.595, "WZTo3LNu_amcNLO": 4.42965, "WZTo1L1Nu2Q_amcNLO": 10.71, "WW1l1nu2q": 49.997, "WZ1l3nu": 3.05,
						#Single Top Background
						#"Tbar-tchan": 26.23, "T-tchan": 44.07, "Tbar-tW": 35.6, "T-tW": 35.6, 
						"Tbar-tchan": 80.8, "T-tchan": 134.2, "Tbar-tW": 39.65, "T-tW": 39.65, 
						#Drell-Yan Jets
                        "DYJetsToLL_M-4to50_HT-70to100": 314.8,
                        "DYJetsToLL_M-4to50_HT-100to200": 190.6,
                        "DYJetsToLL_M-4to50_HT-200to400": 42.27,
                        "DYJetsToLL_M-4to50_HT-400to600": 173.8,
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
						"WJetsToLNu_HT-100To200" : 1244.0, "WJetsToLNu_HT-200To400": 337.8, "WJetsToLNu_HT-400To600": 44.93, "WJetsToLNu_HT-600To800": 11.09, "WJetsToLNu_HT-800To1200": 4.926, "WJetsToLNu_HT-1200To2500" : 1.152, "WJetsToLNu_HT-2500ToInf" : 0.02646, 
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
    def __int__(self): #Additional arguements can be added later
        pass

    def process(self, events):
        vector.register_awkward()


if __name__ == "__main__":
    print("Test Stuff")

    #Diretory for files
    Skimmed_Ganesh_base = "root://cmsxrootd.hep.wisc.edu//store/user/gparida/HHbbtt/Hadded_Skimmed_Files/Full_Production_CMSSW_13_0_13_Nov24_23/LooseSelection_MET_gt_80_nFatJet_gt_0_Skim/2018/"
    
    file_dict = {
            "TTToSemiLeptonic": list(np.append([Skimmed_Ganesh_base + "TTToSemiLeptonic_" + str(j) + ".root" for j in range(2,5)], Skimmed_Ganesh_base + "TTToSemiLeptonic.root")),
            "TTTo2L2Nu": [Skimmed_Ganesh_base + "TTTo2L2Nu.root", Skimmed_Ganesh_base + "TTTo2L2Nu_2.root"],
            "TTToHadronic": [Skimmed_Ganesh_base + "TTToHadronic.root"],
            "ZZ4l": [Skimmed_Ganesh_base + "ZZTo4L.root"],
            "VV2l2nu": [Skimmed_Ganesh_base + "WWTo2L2Nu.root"],
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
            "MET_Data": [Skimmed_Ganesh_base + "/"]

