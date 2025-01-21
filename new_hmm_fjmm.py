import uproot
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import time
import os,sys
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorchtools import *
from torch.optim.lr_scheduler import StepLR

from sklearn.preprocessing import normalize, MinMaxScaler, RobustScaler, QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE

def parse_int_list(value):
    return [int(x) for x in value.split(',')]

parser = argparse.ArgumentParser(description='input parameters to the training')
parser.add_argument('--traincsv', type=str, help='input csv')
parser.add_argument('--epochs', type=int, default=10, help='The epochs of training')
parser.add_argument('--batches', type=int, default=128, help='The batched of training')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--dp', type=float, default=0.1, help='Drop out')
parser.add_argument('--outputmodel', type=str, default='test_model_fjmm.pt', help='The saved DNN model')
parser.add_argument('--outputdir', type=str, default='./', help='The saved dir')
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
os.system('mkdir -p {}'.format(args.outputdir))

# record the start time
start_time = time.time()

# load the input csv file
df = pd.read_csv(args.traincsv)

# define the signal and background dataframe
vhmm_df = df[(df["is_vhmm"] == 1)]
fjmm_bkg_df = df[(df["is_vhmm"] == 0) & (df["is_data"] == 0)]
# fjmm_bkg_df = df[ (df["is_dyjets"] == 1)]
print(vhmm_df.describe())
print(fjmm_bkg_df.describe())

# Tag the label
vhmm_df.loc[:,'isSig'] = True
fjmm_bkg_df.loc[:,'isSig'] = False

sigbkg_df = pd.concat([vhmm_df, fjmm_bkg_df],axis=0)
data = sigbkg_df

# Define input and target variables
# corresponds X and Y
train_var = [
    # #new variable
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
X = data[train_var].to_numpy(dtype=np.float32)
y = data["is_vhmm"].to_numpy(dtype=np.float32)

# feature scaling
# Define scaler to transform input variables
scaler = RobustScaler()
X = scaler.fit_transform(X)
print("Checking the feature scaling...")
print("After scaling, the first variable {}'s values is :".format(train_var[0]))
print(X[:,0])

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

early_stopping = SecondEarlyStopping(patience=10, min_delta=0.004, n_epochs=7)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# if device == 'cuda':
#     torch.cuda.empty_cache()

# define the model, criterion and optimizer
model = CustomDynamicNet(input_size=X.shape[1], hidden_nodes=args.int_array, num_layers=len(args.int_array)).to(device)
# model = nn.DataParallel(model)
# model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
sch_index = True

# Split the data into train and validation sets
batch_size = args.batches
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Create TensorDataset for train and validation sets
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
# Create DataLoader for train and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# training
val_auc_scores = []
avg_train_losses = []
avg_val_losses = []
num_epochs = args.epochs
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for batch_inputs, batch_labels in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        
        train_outputs = model(batch_inputs)
        train_loss = criterion(train_outputs, batch_labels.view(-1, 1))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())
    # Calculate average train loss per epoch
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_train_losses.append(avg_train_loss)
    
    model.eval()
    val_losses = []
    val_predictions = []
    val_targets = []
    for batch_inputs, batch_labels in val_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        
        val_outputs = model(batch_inputs)
        val_loss = criterion(val_outputs, batch_labels.view(-1, 1))
        val_losses.append(val_loss.item())
        val_predictions.append(val_outputs.detach().cpu().numpy())
        val_targets.append(batch_labels.detach().cpu().numpy())

    # Calculate average val loss per epoch
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_val_losses.append(avg_val_loss)

    early_stopping(avg_train_loss, avg_val_loss)

    # Concatenate predictions and targets for the entire validation set
    val_predictions = np.concatenate(val_predictions)
    val_targets = np.concatenate(val_targets)
    # calculate AUC for validation set
    fpr, tpr, thresholds = roc_curve(val_targets, val_predictions)
    val_auc = auc(fpr, tpr)
    val_auc_scores.append(val_auc)
    
    # if epoch == 0 :
    #     print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch, num_epochs, avg_train_loss, avg_val_loss))
    # if (epoch+1) % 50 == 0:
    #     print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, avg_train_loss, avg_val_loss))
    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, avg_train_loss, avg_val_loss))
    if early_stopping.should_stop:
        print("early stopping")
        sch_index = False
        break
    
    if sch_index:
        scheduler.step()
    else:
        pass
    # scheduler.step()
early_stopping.reset()
# save the loss plots
plt.figure()
plt.plot(avg_train_losses, label='Train Loss')
plt.plot(avg_val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(args.outputdir+'/'+'Loss.png')

# save the ROC curves
AUC = 0
for i in range(len(fpr)-1):
    AUC=AUC+(fpr[i+1]-fpr[i])*(tpr[i]+tpr[i+1])
auc=0.5*AUC
print("AUC : {}".format(auc))

plt.figure()
plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), '--', label='random')
plt.plot(fpr, tpr, label='DNN (AUC = {:.4f})'.format(val_auc_scores[-1]))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig(args.outputdir+'/'+'ROC_curves.png')

# make a class prediction for one row of data
def predict(row, model):
    model.eval()
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.cpu().detach().numpy()
    return yhat

train_inputs = torch.from_numpy(X_train).float()
val_inputs = torch.from_numpy(X_val).float()
train_inputs, val_inputs = train_inputs.to(device), val_inputs.to(device)

# Get output scores for training and test data
train_scores = predict(train_inputs, model)
train_scores_sig = train_scores[y_train == 1]
train_socres_bkg = train_scores[y_train == 0]

val_scores = predict(val_inputs, model)
val_scores_sig = val_scores[y_val == 1]
val_scores_bkg = val_scores[y_val == 0]

# Create histograms of output scores
decisions = []
decisions += [train_scores_sig,train_socres_bkg,val_scores_sig,val_scores_bkg]

low = min(np.min(d) for d in decisions)
high = max(np.max(d) for d in decisions)
low_high = (low,high)
bins = 20

plt.figure()
# train sig
sig_train,bins,patches = plt.hist(train_scores_sig,
         color='r', alpha=0.5, range=low_high, bins=bins,
         histtype='stepfilled', density=True,
         label='signal (train)'
)
# train bkg
bkg_train,bins,patches = plt.hist(train_socres_bkg,
         color='b', alpha=0.5, range=low_high, bins=bins,
         histtype='stepfilled', density=True,
         label='background (train)'
)
# val sig
hist, bins = np.histogram(val_scores_sig, bins=bins, range=low_high, density=True)
scale = len(val_scores_sig) / sum(hist)
err   = np.sqrt(hist * scale) / scale
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='signal (val)')
# val bkg
hist, bins = np.histogram(val_scores_bkg, bins=bins, range=low_high, density=True)
scale = len(val_scores_bkg) / sum(hist)
err   = np.sqrt(hist * scale) / scale
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='background (val)')
plt.legend(loc='best',frameon=True,edgecolor='blue',facecolor='blue') 
plt.yscale('log')
plt.legend()
plt.savefig(args.outputdir+'/'+'Score_Distribution.png')

plt.figure()
# train sig
sig_train,bins,patches = plt.hist(train_scores_sig,
         color='r', alpha=0.5, range=low_high, bins=bins,
         histtype='stepfilled', density=True,
         label='signal (train)'
)
# train bkg
bkg_train,bins,patches = plt.hist(train_socres_bkg,
         color='b', alpha=0.5, range=low_high, bins=bins,
         histtype='stepfilled', density=True,
         label='background (train)'
)
# val sig
hist, bins = np.histogram(val_scores_sig, bins=bins, range=low_high, density=True)
scale = len(val_scores_sig) / sum(hist)
err   = np.sqrt(hist * scale) / scale
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='signal (val)')
# val bkg
hist, bins = np.histogram(val_scores_bkg, bins=bins, range=low_high, density=True)
scale = len(val_scores_bkg) / sum(hist)
err   = np.sqrt(hist * scale) / scale
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='background (val)')
plt.legend(loc='best',frameon=True,edgecolor='blue',facecolor='blue') 
# plt.yscale('log')
plt.legend()
plt.savefig(args.outputdir+'/'+'Score_Distribution_linear.png')

# Save the model
torch.save(model.state_dict(), args.outputdir+'/'+args.outputmodel)
print("Save Done!")
print(f"Training spent time: {time.time() - start_time} seconds")