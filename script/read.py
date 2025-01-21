import torch  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load('/home/olympus/MingxuanZhang/fatjet/output_div/234/1_output006/h/test_model_fjmm.pt')
# m = state_dict().to(device)
# print(m)

for layer_name, params in state_dict.items():
    print(f"Layer: {layer_name} | Size: {params.size()}")

