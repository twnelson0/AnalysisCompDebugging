import uproot

if __name__ == "__main__":
    print("Test")
    
    with uproot.open("/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/ZZTo4L_26August25_0757_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_99.root") as f:
        print(f["Events"]["boostedTau_pt"].array())



