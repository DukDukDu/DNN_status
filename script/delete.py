import pandas as pd


# file_path = '/home/olympus/MingxuanZhang/fatjet/vh_ggh/csv/apply4_fjmm.csv'  
# df = pd.read_csv(file_path)


# # df = df.drop(columns=["pnetW_ov_pnetQCD", "pnetZ_ov_pnetQCD"])

# # df.to_csv('apply4.csv', index=False) 

# df['w_ov_QCD'] = df['fatjet_PNet_withMass_WvsQCD']/(df['fatjet_PNet_withMass_WvsQCD'] +df['fatjet_PNet_withMass_QCD'])
# df['z_ov_QCD'] = df['fatjet_PNet_withMass_ZvsQCD']/(df['fatjet_PNet_withMass_ZvsQCD']+ df['fatjet_PNet_withMass_QCD'])
# df['ptmu1_ov_pt2mu'] = df['mu1_fromH_pt']/df['H_pt']
# df['ptmu2_ov_pt2mu'] = df['mu2_fromH_pt']/df['H_pt']
# df['ptmu1_ov_m2mu'] = df['mu1_fromH_pt']/df['H_mass']
# df['ptmu2_ov_m2mu'] = df['mu2_fromH_pt']/df['H_mass']
# df['pt2mu_ov_m2mu'] = df['H_pt']/df['H_mass']
# df['ptfj_ov_pt2mu'] = df['fatjet_pt']/df['H_pt']
# df['ptfj_ov_m2mu'] = df['fatjet_pt']/df['H_mass']
# df['mfj_ov_m2mu'] = df['fatjet_mass']/df['H_mass']
# df['msdfj_ov_m2mu'] = df['fatjet_msoftdrop']/df['H_mass']
# df['pnetW_ov_pnetTWZQCD'] = df['fatjet_PNet_withMass_WvsQCD']/(df['fatjet_PNet_withMass_QCD']+df['fatjet_PNet_withMass_TvsQCD']+df['fatjet_PNet_withMass_WvsQCD']+df['fatjet_PNet_withMass_ZvsQCD'])
# df['pnetZ_ov_pnetTWZQCD'] = df['fatjet_PNet_withMass_ZvsQCD']/(df['fatjet_PNet_withMass_QCD']+df['fatjet_PNet_withMass_TvsQCD']+df['fatjet_PNet_withMass_WvsQCD']+df['fatjet_PNet_withMass_ZvsQCD'])
# df['pnet_score_ratio'] = (df['fatjet_PNet_withMass_WvsQCD'] + df['fatjet_PNet_withMass_ZvsQCD'])/(df['fatjet_PNet_withMass_WvsQCD']+df['fatjet_PNet_withMass_ZvsQCD']+df['fatjet_PNet_withMass_QCD'])

# df.to_csv('/home/olympus/MingxuanZhang/fatjet/vh_ggh/csv/apply4.csv', index=False) 

#divide file to 3 parts
# new_f_path = "train_124.csv"
# file_id = '124'
# n_df = pd.read_csv(new_f_path)

# df_0_5 = n_df[n_df['pnet_score_ratio']<0.5]
# df_5_8 = n_df[(n_df['pnet_score_ratio']>=0.5) & (n_df['pnet_score_ratio']<0.8)]
# df_8_1 = n_df[n_df['pnet_score_ratio']>=0.8]

# df_0_5.to_csv('./low/train{0}_l.csv'.format(file_id), index = False)
# df_5_8.to_csv('./medium/train{0}_m.csv'.format(file_id), index = False)
# df_8_1.to_csv('./high/train{0}_h.csv'.format(file_id), index = False)

## make signal and bkg balance
file_id = '234'
nn_f_path = "./high/train{0}_h_balence.csv".format(file_id)
nn_df = pd.read_csv(nn_f_path)
print(nn_df[nn_df['is_vhmm']==True].shape[0])
print(nn_df[nn_df['is_vhmm']==False].shape[0])

# df_true = nn_df[nn_df['is_vhmm'] == False]
# df_combined = pd.concat([nn_df, df_true], ignore_index=True)
# df_combined = pd.concat([df_combined, df_true], ignore_index = True)
# df_combined.to_csv('./high/train{0}_h_balence.csv'.format(file_id), index=False)
