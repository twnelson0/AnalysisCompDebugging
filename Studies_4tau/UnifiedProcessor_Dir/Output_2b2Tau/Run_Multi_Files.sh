#!/bin/bash
for nTau in {0..4}; do
	python3 Make_Coffea_Plots.py -f "output_""$nTau""_boosted_tau_selec_SingleMuData_QCD_2b2TauSamples_WithSingleMuTrigger_HTC_METPhiCorrections.coffea"  -n $nTau	
done
