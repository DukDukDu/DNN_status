import ROOT


f = ROOT.TFile("/home/olympus/MingxuanZhang/fatjet/output/123/apply03/output.root", "READ")
tree = f.Get("tree")


out = ROOT.TFile("/home/olympus/MingxuanZhang/fatjet/output/123/apply03/run03.root", "RECREATE")
folder = out.mkdir("VH")
folder.cd()

pileup = 'puweight*' +\
         'id_wgt_mu_1*id_wgt_mu_2*iso_wgt_mu_1*iso_wgt_mu_2*' +\
         'id_wgt_mu_1_below15*id_wgt_mu_2_below15*iso_wgt_mu_1_below15*iso_wgt_mu_2_below15*'
cut = '( (trg_single_mu24==1&&is_2017==0)||(trg_single_mu27==1&&is_2017==1) )'


vhmm = ROOT.TH1F("vhmm", " ", 100, 0, 1)
bkg = ROOT.TH1F("bkg", " ", 100, 0, 1)
data_obs = ROOT.TH1F("data_obs", " ", 100, 0, 1)
vhmm.Sumw2()
bkg.Sumw2()
data_obs.Sumw2()


#c1 = ROOT.TCanvas()
tree.Draw('dnn_fjmm_score >> bkg','( ( ((is_dyjets==0)&&(is_diboson==1||is_triboson==1||is_top==1)) || \
            (is_dyjets==1) ) && ({0}) )*Train_weight*{1}*(1 + 4*(1-is_dyjets))'.format(cut,pileup),'goff')
#c1.SaveAs("./apply11/bkg.png")


#c2 = ROOT.TCanvas()
tree.Draw('dnn_fjmm_score >> vhmm','( (is_vhmm==1 && is_dyjets==0) && ({0}) )*Train_weight*{1}*4'.format(cut,pileup),'goff')
#c2.SaveAs("./apply11/sig.png")
tree.Draw('dnn_fjmm_score >> data_obs','( ( is_data==1 && is_dyjets==0 ) && ({0}) )*Train_weight*{1}*4'.format(cut,pileup),'goff')

bkg.Write()
vhmm.Write()
data_obs.Write()


out.Close()


# sig_h_score = ROOT.TH1F("sig_h_score", " ", 80, 110, 150)
# bkg_h_score = ROOT.TH1F("bkg_h_score", " ", 80, 110, 150)
# sig_l_score = ROOT.TH1F("sig_l_score", " ", 80, 110, 150)
# bkg_l_score = ROOT.TH1F("bkg_l_score", " ", 80, 110, 150)


# c3 = ROOT.TCanvas("c3", " ", 800, 600)
# sig_l_score.SetTitle("dnn_score <= 0.7")
# tree.Draw("H_mass>>sig_l_score", "dnn_diboson_score<=0.7 && is_vhmm==1")
# tree.Draw("H_mass>>bkg_l_score", "dnn_diboson_score<=0.7 && (( ( ((is_driven_dy==0)&&(is_diboson==1||is_triboson==1||is_top==1)) \
#     || (is_driven_dy==1) ) && (( (trg_single_mu24==1&&is_2017==0)||(trg_single_mu27==1&&is_2017==1) )\
#     &&Flag_dimuon_Zmass_veto==1) )*Train_weight*wz_zz_scale*(1 + 4*(1-is_driven_dy)))")
# sig_l_score.Scale(1/sig_l_score.Integral())
# sig_l_score.SetLineColor(ROOT.kRed)
# sig_l_score.SetStats(0)
# sig_l_score.Draw("hist")
# bkg_l_score.Scale(1/bkg_l_score.Integral())
# bkg_l_score.SetLineColor(ROOT.kBlack)
# bkg_l_score.Draw("same hist")

# legend = ROOT.TLegend(0.7, 0.8, 0.9, 0.9)
# legend.AddEntry(sig_l_score, "signal", "l")
# legend.AddEntry(bkg_l_score, "bkg", "l")
# legend.Draw()
# c3.SaveAs("./apply11/H_mass_l_score.png")


# c4 = ROOT.TCanvas("c4", " ", 800, 600)
# sig_h_score.SetTitle("dnn_score > 0.7")
# tree.Draw("H_mass>>sig_h_score", "dnn_diboson_score>0.7 && is_vhmm==1")
# tree.Draw("H_mass>>bkg_h_score", "dnn_diboson_score>0.7 && (( ( ((is_driven_dy==0)&&(is_diboson==1||is_triboson==1||is_top==1)) \
#     || (is_driven_dy==1) ) && (( (trg_single_mu24==1&&is_2017==0)||(trg_single_mu27==1&&is_2017==1) )\
#     &&Flag_dimuon_Zmass_veto==1) )*Train_weight*wz_zz_scale*(1 + 4*(1-is_driven_dy)))")
# sig_h_score.Scale(1/sig_h_score.Integral())
# sig_h_score.SetLineColor(ROOT.kRed)
# sig_h_score.SetStats(0)
# sig_h_score.Draw("hist")
# bkg_h_score.Scale(1/bkg_h_score.Integral())
# bkg_h_score.SetLineColor(ROOT.kBlack)
# bkg_h_score.Draw("same hist")

# legend0 = ROOT.TLegend(0.7, 0.8, 0.9, 0.9)
# legend0.AddEntry(sig_h_score, "signal", "l")
# legend0.AddEntry(bkg_h_score, "bkg", "l")
# legend0.Draw()
# c4.SaveAs("./apply11/H_mass_h_score.png")
