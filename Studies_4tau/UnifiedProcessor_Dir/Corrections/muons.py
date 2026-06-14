from pathlib import Path
import awkward as ak
import correctionlib
from correctionlib.schemav2 import Correction, CorrectionSet
import gzip
import numpy as np

def Muon_ID(era):
    if (era == "2018"):
        cval = correctionlib.CorrectionSet.from_file("Corrections/muonefficiencies/Run2/UL/2018/2018_highPt/ScaleFactors_Muon_highPt_IDISO_2018_schemaV2.json")
    else:
        raise Exception("Error: Unkown era %s"%era)
    
    return cval

def Muon_Trigger(era):
    if (era == "2018"):
        cval = correctionlib.CorrectionSet.from_file("Corrections/muonefficiencies/Run2/UL/2018/2018_highPt/ScaleFactors_Muon_highPt_HLT_2018_schemaV2.json")
    else:
        raise Exception("Error: Unkown era %s"%era)
    
    return cval
