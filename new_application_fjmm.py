import os
import uproot
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from sklearn.preprocessing import normalize, MinMaxScaler, RobustScaler, QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

def parse_int_list(value):
    return [int(x) for x in value.split(',')]

parser = argparse.ArgumentParser(description='input parameters to the apply')
parser.add_argument('--inputmodel', type=str, default='test_model_fjmm.pt', help='The input DNN model')
parser.add_argument('--inputcsv', type=str, default='apply_ml_2022preEE_fjmm.csv', help='input csv (preEE or postEE)')
parser.add_argument('--dp', type=float, default=0.1, help='Drop out')
parser.add_argument('--outputdir', type=str, default='./', help='Output Dir')
parser.add_argument('--outputcsv', type=str, default='output_fjmm.csv', help='The saved csv')
parser.add_argument(
    '--int_array',      
    type=parse_int_list, 
    required=True,       
    help="Enter a list of integers separated by commas (e.g., 1,2,3,4)."
)
# Parse the input arguments
args = parser.parse_args()
print("The input parameters are:")
print(args)

class CustomDynamicNet(nn.Module):
    def __init__(self, input_size, hidden_nodes, num_layers):
        
        super(CustomDynamicNet, self).__init__()
        
        assert len(hidden_nodes) == num_layers, "Length of hidden_nodes must match num_layers."
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_size, hidden_nodes[0]))
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_nodes[0])])
        
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_nodes[i-1], hidden_nodes[i]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_nodes[i]))
        
        self.output_layer = nn.Linear(hidden_nodes[-1], 1)
        self.dropout = nn.Dropout(p=args.dp)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x))  
            x = self.batch_norms[i](x)
            x = self.dropout(x)  
        
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

import time
start_time = time.time()
df = pd.read_csv(args.inputcsv)

train_var = [
    #new variable
    "ptmu1_ov_pt2mu", "ptmu2_ov_pt2mu", "ptmu1_ov_m2mu", "ptmu2_ov_m2mu",\
    "pt2mu_ov_m2mu", "ptfj_ov_pt2mu", "ptfj_ov_m2mu", "mfj_ov_m2mu",\
    "msdfj_ov_m2mu", "pnetW_ov_pnetTWZQCD",\
    "pnetZ_ov_pnetTWZQCD","w_ov_QCD", "z_ov_QCD",\

    "H_pt", "H_eta", "H_phi",\
    "mumuH_dR",\
        
    "H_mass",\
    "mu1_fromH_eta", "mu1_fromH_phi", "mu1_fromH_pt",\
    "mu2_fromH_eta", "mu2_fromH_phi", "mu2_fromH_pt",\
    "mu1_mu2_dphi",\
    "met_pt", "met_phi", "met_H_dphi",\
            
    "fatjet_pt", "fatjet_eta", "fatjet_phi",\
    "fatjet_mass", "fatjet_msoftdrop",\
    "fatjet_mmH_dR", "fatjet_mmH_deta", "fatjet_mmH_dphi",\
    "fatjet_mu1_dR", "fatjet_mu1_deta", "fatjet_mu1_dphi",\
    "fatjet_mu2_dR", "fatjet_mu2_deta", "fatjet_mu2_dphi",\
    "fatjet_PNet_withMass_QCD", "fatjet_PNet_withMass_TvsQCD",\
    "fatjet_PNet_withMass_WvsQCD", "fatjet_PNet_withMass_ZvsQCD",\
]

data = pd.DataFrame(df)
X = data[train_var].to_numpy(dtype=np.float32)
scaler = RobustScaler()
X = scaler.fit_transform(X)
X = torch.from_numpy(X).float()
X = X.to(device)
model = CustomDynamicNet(input_size=X.shape[1], hidden_nodes=args.int_array, num_layers=len(args.int_array)).to(device)
model.load_state_dict(torch.load(args.inputmodel))

model.eval()
y_pred = model(X)
y_pred = y_pred.cpu().detach().numpy()
data['dnn_fjmm_score'] = y_pred
data.to_csv(args.outputdir+'/'+args.outputcsv, index=False)

# plot log
plt.figure()
score = data['dnn_fjmm_score']
sig = score[(data['is_vhmm'] == True)]
bkg = score[(data['is_vhmm'] == False) & (data['is_data'] == False)]
# bkg = score[(data['is_dyjets'] == True)]
plt.hist(sig, bins=20, alpha=0.5, range=(0,1), label="Signal", color="red", density=True)
plt.hist(bkg, bins=20, alpha=0.5, range=(0,1), label="bkg", color="blue", density=True)
plt.xlabel('DNN fjmm score')
plt.ylabel("Normalized to Unity")
plt.legend()
plt.yscale('log')
prefix = os.path.splitext(args.outputcsv)[0]
plt.savefig(args.outputdir+'/'+prefix+'_apply_distribution.png')

# plot linear
plt.figure()
score = data['dnn_fjmm_score']
sig = score[(data['is_vhmm'] == True)]
bkg = score[(data['is_vhmm'] == False) & (data['is_data'] == False)]
plt.hist(sig, bins=20, alpha=0.5, range=(0,1), label="Signal", color="red", density=True)
plt.hist(bkg, bins=20, alpha=0.5, range=(0,1), label="bkg", color="blue", density=True)
plt.xlabel('DNN fjmm score')
plt.ylabel("Normalized to Unity")
plt.legend()
prefix = os.path.splitext(args.outputcsv)[0]
plt.savefig(args.outputdir+'/'+prefix+'_apply_distribution_linear.png')

print("Apply Done!")