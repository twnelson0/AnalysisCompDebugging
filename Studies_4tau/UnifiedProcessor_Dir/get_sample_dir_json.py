import json
import glob
import numpy as np

if __name__ == "__main__":
	#Diretory for files
	Skimmed_4tau_base_MC = "root://cmsxrootd.hep.wisc.edu//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/"
	Skimmed_4tau_base_Data = "root://cmsxrootd.hep.wisc.edu//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/Data/"
	Skimmed_4tau_loc_Data = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/Data/"
	Skimmed_4tau_loc_MC = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/"
	
	#Make full arrays of single Muon data
	SingleMu_2018A = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim") 
	SingleMu_2018B = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018B_15January26_0731_skim_Jan26Skim") 
	SingleMu_2018C = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018C_15January26_0740_skim_Jan26Skim") 
	SingleMu_2018D = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018D_15January26_0815_skim_Jan26Skim") 
	
	TTToSemiLeptonic_2018 = glob.glob(Skimmed_4tau_loc_MC + "TTToSemiLeptonic_35August25_0448_skim_Newskim")
	TTTo2L2Nu_2018 = glob.glob(Skimmed_4tau_loc_MC + "TTTo2L2Nu_26August25_0719_skim_Newskim")
	TTToHadronic_2018 = glob.glob(Skimmed_4tau_loc_MC + "TTToHadronic_25October25_0813_skim_Newskim")
	ZZ4L_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo4L_26August25_0757_skim_Newskim")
	ZZTo2L2Nu_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo2L2Nu_04March26_0503_skim_Newskim")
	ZZTo2L2Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo2Q2L_26August25_1034_skim_Newskim")
	ZZTo2Nu2Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo2Nu2Q_04March26_0510_skim_Newskim")
	ZZTo4Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo4Q_04March26_0505_skim_Newskim")
	VV2l2nu_2018 = glob.glob(Skimmed_4tau_loc_MC + "WWTo2L2Nu_26August25_1040_skim_Newskim")
	WWTo1L1Nu2Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "WWTo2L2Nu_26August25_1040_skim_Newskim")
	WWTo4Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "WWTo4Q_04March26_0512_skim_Newskim")
	WZ1l3nu_2018 = glob.glob(Skimmed_4tau_loc_MC + "WZTo1L3Nu_4f_26August25_1016_skim_Newskim")
	ZZ2l2q_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo2Q2L_26August25_1034_skim_Newskim")
	WZ2l2q_2018 = glob.glob(Skimmed_4tau_loc_MC + "WZTo2L2Q_26August25_0926_skim_Newskim")
	WZ1l1nu2q_2018 = glob.glob(Skimmed_4tau_loc_MC + "WZTo1L1Nu2Q_26August25_0840_skim_Newskim")
	DYJetsToLL_M4to50_HT70to100_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-70to100_12December25_1606_skim_Oldskim")
	DYJetsToLL_M4to50_HT100to200_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-100to200_12December25_1604_skim_Oldskim")
	DYJetsToLL_M4to50_HT200to400_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-200to400_12December25_1544_skim_Oldskim")
	DYJetsToLL_M4to50_HT400to600_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-400to600_12December25_1552_skim_Oldskim")
	DYJetsToLL_M4to50_HT600toInf_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-600toInf_12December25_1608_skim_Oldskim")
	DYJetsToLL_M50_HT70to100_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-70to100_12December25_1556_skim_Oldskim")
	DYJetsToLL_M50_HT100to200_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-100to200_12December25_1548_skim_Oldskim")
	DYJetsToLL_M50_HT200to400_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-200to400_12December25_1559_skim_Oldskim")
	DYJetsToLL_M50_HT400to600_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-400to600_12December25_1546_skim_Oldskim")
	DYJetsToLL_M50_HT600to800_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-600to800_12December25_1555_skim_Oldskim")
	DYJetsToLL_M50_HT800to1200_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-800to1200_12December25_1602_skim_Oldskim")
	DYJetsToLL_M50_HT1200to2500_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-1200to2500_12December25_1547_skim_Oldskim")
	DYJetsToLL_M50_HT2500toInf_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-1200to2500_12December25_1547_skim_Oldskim")
	Ttchan_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_t-channel_top_4f_InclusiveDecays_26August25_0843_skim_Newskim")
	Tbartchan_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_t-channel_antitop_4f_InclusiveDecays_26August25_0821_skim_Newskim")
	TtW_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_tW_top_5f_inclusiveDecays_26August25_0753_skim_Newskim")
	TbartW_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_tW_antitop_5f_inclusiveDecays_26August25_1030_skim_Newskim")
	ST_schannel_4f_hadronicDecays_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_s-channel_4f_hadronicDecays_04March26_0506_skim_Newskim")
	ST_schannel_4f_leptonDecays_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_s-channel_4f_leptonDecays_04March26_0507_skim_Newskim")
	WJetsToLNu_HT70To100_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-70To100_04March26_0515_skim_Newskim")
	WJetsToLNu_HT100To200_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-100To200_26August25_0810_skim_Newskim")
	WJetsToLNu_HT200To400_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-200To400_26August25_0709_skim_Newskim")
	WJetsToLNu_HT400To600_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-400To600_26August25_1014_skim_Newskim")
	WJetsToLNu_HT400To600_2018_Other = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-400To600_OtherPart_26August25_1032_skim_Newskim")
	WJetsToLNu_HT600To800_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-600To800_26August25_0755_skim_Newskim")
	WJetsToLNu_HT600To800_2018_Other = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-600To800_OtherPart_26August25_0752_skim_Newskim")
	WJetsToLNu_HT800To1200_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-800To1200_26August25_0708_skim_Newskim")
	WJetsToLNu_HT800To1200_2018_Other = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-800To1200_OtherPart_26August25_0925_skim_Newskim")
	WJetsToLNu_HT1200To2500_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-1200To2500_26August25_1016_skim_Newskim")
	WJetsToLNu_HT1200To2500_2018_Other = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-1200To2500_OtherPart_26August25_1041_skim_Newskim")
	WJetsToLNu_HT2500ToInf_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-2500ToInf_26August25_1047_skim_Newskim")
	WJetsToLNu_HT2500ToInf_2018_Other = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-2500ToInf_OtherPart_26August25_1043_skim_Newskim")
	QCD_HT50To100 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT50to100_23April26_0525_skim_FourTauSkim")
	QCD_HT100To200 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT100to200_23April26_0519_skim_FourTauSkim")
	QCD_HT200To300 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT200to300_23April26_0542_skim_FourTauSkim")
	QCD_HT300To500 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT300to500_23April26_0555_skim_FourTauSkim")
	QCD_HT500To700 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT500to700_23April26_0512_skim_FourTauSkim")
	QCD_HT700To1000 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT700to1000_23April26_0528_skim_FourTauSkim")
	QCD_HT1000To1500 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT1000to1500_23April26_0536_skim_FourTauSkim")
	QCD_HT1500To2000 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT1500to2000_23April26_0539_skim_FourTauSkim")
	QCD_HT2000ToInf = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT2000toInf_23April26_0541_skim_FourTauSkim")
	
	file_dict_full = {
			"TTToSemiLeptonic": [file for file in TTToSemiLeptonic_2018][0],
			"TTTo2L2Nu": [file for file in TTTo2L2Nu_2018][0],
			"TTToHadronic": [file for file in TTToHadronic_2018][0],
			"ZZ4l": [file for file in ZZ4L_2018][0],
			"ZZTo2L2Nu": [file for file in ZZTo2L2Nu_2018][0],
			"ZZTo2Nu2Q": [file for file in ZZTo2Nu2Q_2018][0],
			"VV2l2nu": [file for file in VV2l2nu_2018][0],
			"ZZTo4Q" : [file for file in ZZTo4Q_2018][0],
			"WWTo1L1Nu2Q": [file for file in WWTo1L1Nu2Q_2018][0],
			"WWTo4Q": [file for file in WWTo4Q_2018][0],
			"WZ1l3nu": [file for file in WZ1l3nu_2018][0],
			"ZZ2l2q": [file for file in ZZ2l2q_2018][0],
			"WZ2l2q": [file for file in WZ2l2q_2018][0],
			"WZ1l1nu2q" : [file for file in WZ1l1nu2q_2018][0],
			"DYJetsToLL_M-4to50_HT-70to100": [file for file in DYJetsToLL_M4to50_HT70to100_2018][0],
			"DYJetsToLL_M-4to50_HT-100to200": [file for file in DYJetsToLL_M4to50_HT100to200_2018][0],
			"DYJetsToLL_M-4to50_HT-200to400": [file for file in DYJetsToLL_M4to50_HT200to400_2018][0],
			"DYJetsToLL_M-4to50_HT-400to600": [file for file in DYJetsToLL_M4to50_HT400to600_2018][0],
			"DYJetsToLL_M-4to50_HT-600toInf":[file for file in DYJetsToLL_M4to50_HT600toInf_2018][0],
			"DYJetsToLL_M-50_HT-70to100": [file for file in DYJetsToLL_M50_HT70to100_2018][0],
			"DYJetsToLL_M-50_HT-100to200": [file for file in DYJetsToLL_M50_HT100to200_2018][0],
			"DYJetsToLL_M-50_HT-200to400": [file for file in DYJetsToLL_M50_HT200to400_2018][0],
			"DYJetsToLL_M-50_HT-400to600": [file for file in DYJetsToLL_M50_HT400to600_2018][0],
			"DYJetsToLL_M-50_HT-600to800": [file for file in DYJetsToLL_M50_HT600to800_2018][0],
			"DYJetsToLL_M-50_HT-800to1200": [file for file in DYJetsToLL_M50_HT800to1200_2018][0],
			"DYJetsToLL_M-50_HT-1200to2500": [file for file in DYJetsToLL_M50_HT1200to2500_2018][0],
			"DYJetsToLL_M-50_HT-2500toInf": [file for file in DYJetsToLL_M50_HT2500toInf_2018][0],
			"T-tchan": [file for file in Ttchan_2018][0],
			"Tbar-tchan": [file for file in Tbartchan_2018][0],
			"T-tW": [file for file in TtW_2018][0],
			"Tbar-tW": [file for file in TbartW_2018][0],
			"ST_s-channel_4f_hadronicDecays": [file for file in ST_schannel_4f_hadronicDecays_2018][0],
			"ST_s-channel_4f_leptonDecays": [file for file in ST_schannel_4f_leptonDecays_2018][0],
			"WJetsToLNu_HT-70To100": [file for file in WJetsToLNu_HT70To100_2018][0],
			"WJetsToLNu_HT-100To200": [file for file in WJetsToLNu_HT100To200_2018][0],
			"WJetsToLNu_HT-200To400": [file for file in WJetsToLNu_HT200To400_2018][0],
			"WJetsToLNu_HT-400To600": [file for file in WJetsToLNu_HT400To600_2018][0],
			"WJetsToLNu_HT-400To600_Other": [file for file in WJetsToLNu_HT400To600_2018_Other][0],
			"WJetsToLNu_HT-600To800": [file for file in WJetsToLNu_HT600To800_2018][0],
			"WJetsToLNu_HT-600To800_Other": [file for file in WJetsToLNu_HT600To800_2018_Other][0],
			"WJetsToLNu_HT-800To1200": [file for file in WJetsToLNu_HT800To1200_2018][0],
			"WJetsToLNu_HT-800To1200_Other": [file for file in WJetsToLNu_HT800To1200_2018_Other][0],
			"WJetsToLNu_HT-1200To2500": [file for file in WJetsToLNu_HT1200To2500_2018][0],
			"WJetsToLNu_HT-1200To2500_Other": [file for file in WJetsToLNu_HT1200To2500_2018_Other][0],
			"WJetsToLNu_HT-2500ToInf": [file for file in WJetsToLNu_HT2500ToInf_2018][0],
			"WJetsToLNu_HT-2500ToInf_Other": [file for file in WJetsToLNu_HT2500ToInf_2018_Other][0],
			"QCD_HT50to100": [file for file in QCD_HT50To100][0],
			"QCD_HT100to200": [file for file in QCD_HT100To200][0],
			"QCD_HT200to300": [file for file in QCD_HT200To300][0],
			"QCD_HT300to500": [file for file in QCD_HT300To500][0],
			"QCD_HT500to700": [file for file in QCD_HT500To700][0],
			"QCD_HT700to1000": [file for file in QCD_HT700To1000][0],
			"QCD_HT1000to1500": [file for file in QCD_HT1000To1500][0],
			"QCD_HT1500to2000": [file for file in QCD_HT1500To2000][0],
			"QCD_HT2000toInf": [file for file in QCD_HT2000ToInf][0],
			"SingleMuA": [file for file in SingleMu_2018A][0],
			"SingleMuB": [file for file in SingleMu_2018B][0],
			"SingleMuC": [file for file in SingleMu_2018C][0],
			"SingleMuD": [file for file in SingleMu_2018D][0],
		}

	with open("SampleLoc_2018.json","w") as f:
		json.dump(file_dict_full,f, indent=4)
