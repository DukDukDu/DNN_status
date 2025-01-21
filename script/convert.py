import pandas as pd
import ROOT
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='input parameters to the apply')
parser.add_argument('--start', type=int, help='start nb of loop')
parser.add_argument('--end', type=int, help='end nb of loop')
args = parser.parse_args()

def convert(rootfile_name, csvfile_name, columns):
    root_file = ROOT.TFile(rootfile_name, 'RECREATE')
    tree = ROOT.TTree('tree', 'Tree from CSV data')
    branches = {}
    arrays = {}
    for column in columns:
        arrays[column] = np.zeros(1, dtype=np.float32)  
        branches[column] = tree.Branch(column, arrays[column], f'{column}/F')
    
    chunk_size = 10000
    for chunk in pd.read_csv(csvfile_name, chunksize=chunk_size):
        for i, row in chunk.iterrows():
            for column in columns:
                arrays[column][0] = row[column]
            tree.Fill()
    tree.Write('', ROOT.TObject.kOverwrite)
    root_file.Close()

columns = ['ptmu1_ov_pt2mu', 'ptmu2_ov_pt2mu', 'ptmu1_ov_m2mu', 'ptmu2_ov_m2mu', 'pt2mu_ov_m2mu', 'ptfj_ov_pt2mu', 'ptfj_ov_m2mu', 
           'mfj_ov_m2mu', 'msdfj_ov_m2mu', 'pnetW_ov_pnetTWZQCD', 'pnetZ_ov_pnetTWZQCD', 'BosonDecayMode', 'FatJetFlag_pass_veto_map', 
           'Flag_DiMuonFromHiggs', 'Flag_GoodEle_Veto', 'Flag_LeptonChargeSumVeto', 'Flag_MaxMetCut', 'H_eta', 'H_mass', 'H_phi', 'H_pt', 
           'JetFlag_pass_veto_map', 'Train_weight', 'Xsec', 'btag_weight', 'dy_scale', 'event', 'fatjet_PNet_QCD', 'fatjet_PNet_withMass_QCD', 
           'fatjet_PNet_withMass_TvsQCD', 'fatjet_PNet_withMass_WvsQCD', 'fatjet_PNet_withMass_ZvsQCD', 'fatjet_eta', 'fatjet_mass', 
           'fatjet_mmH_dR', 'fatjet_mmH_deta', 'fatjet_mmH_dphi', 'fatjet_msoftdrop', 'fatjet_mu1_dR', 'fatjet_mu1_deta', 'fatjet_mu1_dphi', 
           'fatjet_mu2_dR', 'fatjet_mu2_deta', 'fatjet_mu2_dphi', 'fatjet_phi', 'fatjet_pt', 'genEventSumW', 'genWeight', 'genmet_phi', 'genmet_pt', 
           'id_wgt_mu_1', 'id_wgt_mu_1__MuonIDIsoDown', 'id_wgt_mu_1__MuonIDIsoUp', 'id_wgt_mu_1_below15', 'id_wgt_mu_1_below15__MuonIDIsoDown', 
           'id_wgt_mu_1_below15__MuonIDIsoUp', 'id_wgt_mu_2', 'id_wgt_mu_2__MuonIDIsoDown', 'id_wgt_mu_2__MuonIDIsoUp', 'id_wgt_mu_2_below15', 
           'id_wgt_mu_2_below15__MuonIDIsoDown', 'id_wgt_mu_2_below15__MuonIDIsoUp', 'is_2016', 'is_2017', 'is_2018', 'is_2022', 'is_2023', 'is_TTH_Hto2Mu', 
           'is_TTto2L2Nu', 'is_TWminusto2L2Nu', 'is_TbarWplusto2L2Nu', 'is_WWW', 'is_WWZ', 'is_WWto2L2Nu', 'is_WZZ', 'is_WZto2L2Q', 'is_WZto3LNu', 'is_WminusH_Hto2Mu', 
           'is_WplusH_Hto2Mu', 'is_ZH_Hto2Mu', 'is_ZZZ', 'is_ZZto2L2Nu', 'is_ZZto2L2Q', 'is_ZZto4L', 'is_data', 'is_diboson', 'is_dyjets', 'is_embedding', 
           'is_top', 'is_triboson', 'is_vhmm', 'is_wjets', 'is_zjjew', 'iso_wgt_mu_1', 'iso_wgt_mu_1__MuonIDIsoDown', 'iso_wgt_mu_1__MuonIDIsoUp', 
           'iso_wgt_mu_1_below15', 'iso_wgt_mu_1_below15__MuonIDIsoDown', 'iso_wgt_mu_1_below15__MuonIDIsoUp', 'iso_wgt_mu_2', 'iso_wgt_mu_2__MuonIDIsoDown', 
           'iso_wgt_mu_2__MuonIDIsoUp', 'iso_wgt_mu_2_below15', 'iso_wgt_mu_2_below15__MuonIDIsoDown', 'iso_wgt_mu_2_below15__MuonIDIsoUp', 'lumi', 
           'met_H_dphi', 'met_phi', 'met_pt', 'mu1_MHTALL_dphi', 'mu1_MHT_dphi', 'mu1_fromH_eta', 'mu1_fromH_phi', 'mu1_fromH_pt', 'mu1_mu2_dphi', 
           'mu2_MHTALL_dphi', 'mu2_MHT_dphi', 'mu2_fromH_eta', 'mu2_fromH_phi', 'mu2_fromH_pt', 'mumuH_MHTALL_dphi', 'mumuH_MHT_dphi', 'mumuH_dR', 
           'mumuH_deta', 'mumuH_dphi', 'nbaseelectrons', 'nbasemuons', 'nbjets_loose', 'nbjets_medium', 'nelectrons', 'nfatjets', 'njets', 'nmuons', 
           'puweight', 'puweight__PileUpDown', 'puweight__PileUpUp', 'run', 'smallest_dimuon_mass', 'trg_single_mu24', 'trg_single_mu27', 'trg_wgt_single_mu24', 
           'trg_wgt_single_mu24__singleMuonTriggerSFDown', 'trg_wgt_single_mu24__singleMuonTriggerSFUp', 'w_ov_QCD', 'z_ov_QCD', 'pnet_score_ratio', 'dnn_fjmm_score']

# root_dir = []
# csv_dir = []

# for i in range(args.start, args.end+1):
#     root_dir.append('/home/olympus/MingxuanZhang/fatjet/output_div/234/n_out_app/apply{0:03d}/h/o{0:03d}_h.root'.format(i))
#     csv_dir.append('/home/olympus/MingxuanZhang/fatjet/output_div/234/n_out_app/apply{0:03d}/h/o{0:03d}_h.csv'.format(i))

# for i  in range(args.start, args.end+1):
#     convert(root_dir[i], csv_dir[i], columns)
#     print("convert {0}th file successfully".format(i))

convert('/home/olympus/MingxuanZhang/fatjet/output_div/134/n_out_app/apply069/h/h.root', '/home/olympus/MingxuanZhang/fatjet/output_div/134/n_out_app/apply069/h/o069_h.csv', columns)
