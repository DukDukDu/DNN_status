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

parser = argparse.ArgumentParser(description='input parameters to the apply')
parser.add_argument('--inputmodel', type=str, default='test_model_fjmm.pt', help='The input DNN model')
parser.add_argument('--inputcsv', type=str, default='apply_ml_2022preEE_fjmm.csv', help='input csv (preEE or postEE)')
parser.add_argument('--fc1', type=int, default=1024, help='The first full connected layer size')
parser.add_argument('--fc2', type=int, default=256, help='The second full connected layer size')
parser.add_argument('--fc3', type=int, default=64, help='The third full connected layer size')
parser.add_argument('--dp', type=float, default=0.1, help='Drop out')
parser.add_argument('--outputdir', type=str, default='./', help='Output Dir')
parser.add_argument('--outputcsv', type=str, default='output_fjmm.csv', help='The saved csv')
parser.add_argument('--nl1', type=int, default=1, help='nb of fc1 layers')
parser.add_argument('--nl2', type=int, default=1, help='nb of fc2 layers')
parser.add_argument('--nl3', type=int, default=1, help='nb of fc3 layers')
# Parse the input arguments
args = parser.parse_args()
print("The input parameters are:")
print(args)

class LargeDNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.input_layer = nn.Linear(input_size, args.fc1)
        self.bn_input = nn.BatchNorm1d(args.fc1)  # BatchNorm for input layer
        
        self.hidden_layers1 = nn.ModuleList([
            nn.Linear(args.fc1, args.fc1) for _ in range(args.nl1)
        ])
        self.bn_hidden1 = nn.BatchNorm1d(args.fc1)  # BatchNorm for hidden layer 1
        self.inter_layer1 = nn.Linear(args.fc1, args.fc2)
        self.bn_inter1 = nn.BatchNorm1d(args.fc2)  # BatchNorm for inter layer 1
        
        self.hidden_layers2 = nn.ModuleList([
            nn.Linear(args.fc2, args.fc2) for _ in range(args.nl2)
        ])
        self.bn_hidden2 = nn.BatchNorm1d(args.fc2)  # BatchNorm for hidden layer 2
        self.inter_layer2 = nn.Linear(args.fc2, args.fc3)
        self.bn_inter2 = nn.BatchNorm1d(args.fc3)  # BatchNorm for inter layer 2
        
        self.hidden_layers3 = nn.ModuleList([
            nn.Linear(args.fc3, args.fc3) for _ in range(args.nl3)
        ])
        self.bn_hidden3 = nn.BatchNorm1d(args.fc3)  # BatchNorm for hidden layer 3
        
        self.output_layer = nn.Linear(args.fc3, 1)
        
        self.dropout = nn.Dropout(p=args.dp)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.bn_input(x)  # BatchNorm on input
        x = self.dropout(x)
        
        for layer in self.hidden_layers1:
            x = torch.relu(layer(x))
            x = self.bn_hidden1(x)  # BatchNorm on hidden layer 1
            x = self.dropout(x)
        x = torch.relu(self.inter_layer1(x))
        x = self.bn_inter1(x)  # BatchNorm on inter layer 1
        x = self.dropout(x)
        
        for layer in self.hidden_layers2:
            x = torch.relu(layer(x))
            x = self.bn_hidden2(x)  # BatchNorm on hidden layer 2
            x = self.dropout(x)
        x = torch.relu(self.inter_layer2(x))
        x = self.bn_inter2(x)  # BatchNorm on inter layer 2
        x = self.dropout(x)
        
        for layer in self.hidden_layers3:
            x = torch.relu(layer(x))
            x = self.bn_hidden3(x)  # BatchNorm on hidden layer 3
            x = self.dropout(x)
        
        x = self.output_layer(x)
        x = self.sigmoid(x)  # Sigmoid for binary classification
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
model = LargeDNN(input_size=X.shape[1]).to(device)
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