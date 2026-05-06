#!/bin/bash

Region_Arr=('All' 'TightTCR' 'LooseTCR' 'NotTCR' 'ZCR' 'NotZCR')

for region in "${Region_Arr[@]}"; do
	python3 PlotProducer.py -f output_4_boosted_tau_selec_SingleMuData_4TauSamples_WithSingleMuTrigger_WithQCD_AddRegions.coffea -n 4 -r $region	
done
