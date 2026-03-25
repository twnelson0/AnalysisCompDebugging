#!/bin/bash

#Obtain all possible debugging samples 
for ((i = 0 ; i < 1 ; i++)); do
	debug=$((8 - $i % 9))
	#python3 Run_Debug_Processor.py -f 0 -d $debug
	python3 Run_Debug_Processor.py -f 1 -d $debug
done


