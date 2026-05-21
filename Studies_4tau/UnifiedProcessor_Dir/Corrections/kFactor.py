import hist
import uproot
import numpy as np
from coffea.lookup_tools import extractor

ext = extractor()

#Import kfactor histograms
ext.add_weight_sets(["nlo_EW_WJets EWK_W_nominal Corrections/ZorW_NLO_corrections.root"])
ext.add_weight_sets(["nlo_EW_ZJets EWK_Z_nominal Corrections/ZorW_NLO_corrections.root"])
ext.add_weight_sets(["nlo_QCD_WJets QCD_W Corrections/ZorW_NLO_corrections.root"])
ext.add_weight_sets(["nlo_QCD_ZJets QCD_Z Corrections/ZorW_NLO_corrections.root"])

ext.finalize()
evaluator = ext.make_evaluator()

#k Factor Functions
def getEWKW(pt):
	return evaluator["nlo_EW_WJets"](pt)

def getEWKZ(pt):
	return evaluator["nlo_EW_ZJets"](pt)

def getQCDW(pt):
	return evaluator["nlo_QCD_WJets"](pt)

def getQCDZ(pt):
	return evaluator["nlo_QCD_ZJets"](pt)
