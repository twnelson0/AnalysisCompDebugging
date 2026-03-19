import uproot
import numpy as np
import os
import glob
import time
import json

if __name__ == "__main__":
	numEvents_Dict = {}
	sumWEvents_Dict = {}
	
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
	ZZTo2L2Nu_2018 = glob.glob(Skimmed_4tau_base_MC + "ZZTo2L2Nu_04March26_0503_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZTo2Nu2Q_2018 = glob.glob(Skimmed_4tau_base_MC + "ZZTo2Nu2Q_04March26_0510_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZTo4Q_2018 = glob.glob(Skimmed_4tau_base_MC + "ZZTo4Q_04March26_0505_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	VV2l2nu_2018 = glob.glob(Skimmed_4tau_base_MC + "WWTo2L2Nu_26August25_1040_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WWTo1L1Nu2Q_2018 = glob.glob(Skimmed_4tau_base_MC + "WWTo2L2Nu_26August25_1040_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WWTo4Q_2018 = glob.glob(Skimmed_4tau_base_MC + "WWTo4Q_04March26_0512_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WZ1l3nu_2018 = glob.glob(Skimmed_4tau_base_MC + "WZTo1L3Nu_4f_26August25_1016_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZ2l2q_2018 = glob.glob(Skimmed_4tau_base_MC + "ZZTo2Q2L_26August25_1034_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WZ2l2q_2018 = glob.glob(Skimmed_4tau_base_MC + "WZTo2L2Q_26August25_0926_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WZ1l1nu2q_2018 = glob.glob(Skimmed_4tau_base_MC + "WZTo1L1Nu2Q_26August25_0840_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT70to100_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-4to50_HT-70to100_12December25_1606_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT100to200_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-4to50_HT-100to200_12December25_1604_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT200to400_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-4to50_HT-200to400_12December25_1544_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT400to600_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-4to50_HT-400to600_12December25_1552_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT600toInf_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-4to50_HT-600toInf_12December25_1608_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT70to100_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-50_HT-70to100_12December25_1556_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT100to200_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-50_HT-100to200_12December25_1548_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT200to400_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-50_HT-200to400_12December25_1559_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT400to600_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-50_HT-400to600_12December25_1546_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT600to800_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-50_HT-600to800_12December25_1555_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT800to1200_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-50_HT-800to1200_12December25_1602_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT1200to2500_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-50_HT-1200to2500_12December25_1547_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT2500toInf_2018 = glob.glob(Skimmed_4tau_base_MC + "DYJetsToLL_M-50_HT-1200to2500_12December25_1547_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	Ttchan_2018 = glob.glob(Skimmed_4tau_base_MC + "ST_t-channel_top_4f_InclusiveDecays_26August25_0843_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	Tbartchan_2018 = glob.glob(Skimmed_4tau_base_MC + "ST_t-channel_antitop_4f_InclusiveDecays_26August25_0821_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	TtW_2018 = glob.glob(Skimmed_4tau_base_MC + "ST_tW_top_5f_inclusiveDecays_26August25_0753_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	TbartW_2018 = glob.glob(Skimmed_4tau_base_MC + "ST_tW_antitop_5f_inclusiveDecays_26August25_1030_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ST_schannel_4f_hadronicDecays_2018 = glob.glob(Skimmed_4tau_base_MC + "ST_s-channel_4f_hadronicDecays_04March26_0506_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ST_schannel_4f_leptonDecays_2018 = glob.glob(Skimmed_4tau_base_MC + "ST_s-channel_4f_leptonDecays_04March26_0507_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WJetsToLNu_HT70To100_2018 = glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-70To100_04March26_0515_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WJetsToLNu_HT100To200_2018 = glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-100To200_26August25_0810_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WJetsToLNu_HT200To400_2018 = glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-200To400_26August25_0709_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WJetsToLNu_HT400To600_2018 = np.append(glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-400To600_26August25_1014_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-400To600_OtherPart_26August25_1032_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	WJetsToLNu_HT600To800_2018 = np.append(glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-600To800_26August25_0755_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-600To800_OtherPart_26August25_0752_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	WJetsToLNu_HT800To1200_2018 = np.append(glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-800To1200_26August25_0708_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-800To1200_OtherPart_26August25_0925_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	WJetsToLNu_HT1200To2500_2018 = np.append(glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-120)0To2500_26August25_1016_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-1200To2500_OtherPart_26August25_1041_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	WJetsToLNu_HT2500ToInf_2018 = np.append(glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-2500ToInf_26August25_1047_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_base_MC + "WJetsToLNu_HT-2500ToInf_OtherPart_26August25_1043_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))


	file_dict_data_test = {
		"Data_Mu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in SingleMuA_2018A] 
	}
	
	file_dict_full = {
			"TTToSemiLeptonic": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TTToSemiLeptonic_2018],
			"TTTo2L2Nu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TTTo2L2Nu_2018],
			"TTToHadronic": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TTToHadronic_2018],
			"ZZ4l": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZ4L_2018],
			"ZZTo2L2Nu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZTo2L2Nu_2018],
			"ZZTo2Nu2Q": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZTo2L2Nu_2018],
			"ZZTo4Q": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZTo2L2Nu_2018],
			"VV2l2nu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZTo4Q_2018],
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
	
	#Set file dictionary and list of backgrounds prior to running processor
	file_dict = file_dict_full

	start_time = time.time()
	#print("Start time of code to get sum weights: %d"%start_time)
	for key_name, file_array in file_dict.items(): 
		print(key_name)
		if (key_name != "Data_Mu" and key_name != "Data_MET" ): 
			numEvents_Dict[key_name] = 0 #Initialize the number of events dictionary
			sumWEvents_Dict[key_name] = 0 #Initialize the number of events dictionary
			tempFiles = uproot.dask([file + ":Runs" for file in file_array])
			numEvents_Dict[key_name] += np.sum(tempFiles['genEventCount'].compute()) #Fixed for nanoAOD 
			sumWEvents_Dict[key_name] += np.sum(tempFiles['genEventSumw'].compute()) #Fixed for nanoAOD 
			del tempFiles

		else: #Ignore data files
			numEvents_Dict[key_name] = 1
			sumWEvents_Dict[key_name] = 1
	
	end_time = time.time()
	#print("End time of code to get sum weights: %d"%end_time)
	print("Time to process all files: %d"%(end_time - start_time))

	#Save the sumW and counts as JSON files
	with open("genWeightSum_JSON", "w") as fp:
		json.dump(sumWEvents_Dict, fp)
	
	with open("numEvents_JSON", "w") as fp:
		json.dump(numEvents_Dict, fp)


