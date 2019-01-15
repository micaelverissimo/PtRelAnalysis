import ROOT
from functions import HistogramFunctions,AuxiliarFunctions,FigureFunctions,FitFunctions

import keras
print(keras.__version__)

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from keras.utils import to_categorical
from keras.models import model_from_json
import json

from sklearn.externals import joblib
import pickle

pickle_path = '/home/micael/MyWorkspace/multi_Higgs_classifier/classificadores/'
data_path = '/home/micael/MyWorkspace/multi_Higgs_classifier/data_inputs/'
plots_path = '/home/micael/MyWorkspace/multi_Higgs_classifier/plots_v2/'
ROOT_path = '/home/micael/MyWorkspace/DATA/ROOT_Files/'
bkg_names = ['TTbar', 'Wbb']#, 'Wbl', 'Wll', 'Wcl', 'Wcc', 'WW', 'WZ', 'ZZ']

#signal = np.load(data_path+'lvbb125_data_array.npy')
#signal = signal[:,:23]
#sgn_trgt_unbalanced = np.ones(signal.shape[0], dtype=np.int)

#bkgs  = {}
#targets = {}
#for ibk in bkg_names:
#    print('Creating the dict with {} bkg data and the targets...'.format(ibk))
#    targets[ibk] = {}
#    bkgs[ibk] = np.load(data_path+ibk+'_data_array.npy')
#    bkgs[ibk]  = bkgs[ibk][:,:23]
#    
#    bkg_trgt_balanced = np.zeros(signal.shape[0], dtype=np.int)
#    bkg_trgt_unbalanced = np.zeros(bkgs[ibk].shape[0], dtype=np.int)
#    targets[ibk]['unbalanced'] = np.append(sgn_trgt_unbalanced, bkg_trgt_unbalanced)
#    targets[ibk]['balanced'] = np.append(sgn_trgt_unbalanced, bkg_trgt_balanced)

def get_models(raw_classifiers):
    models = {}
    for ifold in raw_classifiers.keys():
        models[ifold] = {}
        for jinit in raw_classifiers[ifold].keys():
            if jinit == 'best init' or jinit == 'pre_proc':
                continue
            tmp = raw_classifiers[ifold][jinit]['model']
            models[ifold][jinit] = model_from_json(json.dumps(json.loads(tmp), separators=(',',':')))
            models[ifold][jinit].set_weights(raw_classifiers[ifold][jinit]['weights'])
    return models

classifiers = {}
for bkg in bkg_names:
    with open(pickle_path+bkg+'_classificadores.10sorts.10inits.mapstd.mse.subsampling.batchsize1024.earlystopping50.pkl', 'rb') as f:
        u = pickle.load(f)
        classifiers[bkg] = u
classifiers_noPtRel = {}
for bkg in bkg_names:
    with open(pickle_path+bkg+'_classificadores.noPtRel.10sorts.10inits.mapstd.mse.subsampling.batchsize1024.earlystopping50.pkl', 'rb') as f:
        u = pickle.load(f)
        classifiers_noPtRel[bkg] = u

#print('Classifiers: ',classifiers.keys())

#best_inits = {}
#for bkg in classifiers.keys():
#    best_inits[bkg] = {}
#    for ifold in classifiers[bkg][10].keys():
#        best_inits[bkg][ifold] = classifiers[bkg][10][ifold]['best init']
        #print('The best initialization in the {} for {} is : {}'.format(bkg, ifold, classifiers[bkg][10][ifold]['best init']))
#best_inits_noPtRel = {}
#for bkg in classifiers_noPtRel.keys():
#    best_inits_noPtRel[bkg] = {}
#    for ifold in classifiers_noPtRel[bkg][10].keys():
#        best_inits_noPtRel[bkg][ifold] = classifiers_noPtRel[bkg][10][ifold]['best init']
        #print('The best initialization in the {} for {} is : {}'.format(bkg, ifold, classifiers[bkg][10][ifold]['best init']))


models = {}
for imodel in classifiers.keys():
    models[imodel] = get_models(classifiers[imodel][10])

# no PtRel
models_noPtRel = {}
for imodel in classifiers_noPtRel.keys():
    models_noPtRel[imodel] = get_models(classifiers_noPtRel[imodel][10])

#ROOT.gROOT.Reset()
ROOT.gStyle.SetOptStat(0)

# for histograms
bins = 60
m_max = 300
m_min = 0

ylabel = "Occurrences"

# Use Colors
list_color_TTbar = []
list_color_Wbb = []

list_color_TTbar.append(ROOT.kRed) # For Signal
list_color_Wbb.append(ROOT.kRed) # For Signal

list_color_Wbb.append(ROOT.kGreen+3) # For Wbb
list_color_TTbar.append(ROOT.kOrange+0) # For TTbar

#  Make a list of mass histogram
list_inf_M_TTbar = []
list_inf_M_Wbb = []

hist_inf = HistogramFunctions.OneDimHistInfo("Signal Mass ","h_lvbb_M",bins,m_min,m_max,"Invariant Mass",ylabel)
list_inf_M_TTbar.append(hist_inf)
list_inf_M_Wbb.append(hist_inf)


hist_inf = HistogramFunctions.OneDimHistInfo("Bkg Wbb Mass ","h_Wbb_M",bins,m_min,m_max,"Invariant Mass",ylabel)
list_inf_M_Wbb.append(hist_inf)

hist_inf = HistogramFunctions.OneDimHistInfo("Bkg TTbar","h_TTbar_M",bins,m_min,m_max,"Invariant Mass",ylabel)
list_inf_M_TTbar.append(hist_inf)


list_hist_M_TTbar = HistogramFunctions.CreateListOf1DHistograms(list_inf_M_TTbar,list_color_TTbar)
list_hist_M_Wbb = HistogramFunctions.CreateListOf1DHistograms(list_inf_M_Wbb,list_color_Wbb)

list_hist_M_TTbar_noPtRel = HistogramFunctions.CreateListOf1DHistograms(list_inf_M_TTbar,list_color_TTbar)
list_hist_M_Wbb_noPtRel = HistogramFunctions.CreateListOf1DHistograms(list_inf_M_Wbb,list_color_Wbb)

list_hist_M_TTbar_noNN = HistogramFunctions.CreateListOf1DHistograms(list_inf_M_TTbar,list_color_TTbar)
list_hist_M_Wbb_noNN = HistogramFunctions.CreateListOf1DHistograms(list_inf_M_Wbb,list_color_Wbb)

print(list_hist_M_TTbar,list_hist_M_Wbb)


processes = "lvbb125,TTbar,Wbb"

for process in processes.split(","):
    data_input = ROOT_path+process+".root"
    treeName = "perevent"
    file = ROOT.TFile(data_input,"READ")
    tree = file.Get(treeName)
    nrEvents = tree.GetEntries()
    #nrEvents = 10

    for (i,entry) in enumerate(tree):
        if nrEvents>0:
            if i>nrEvents:
                break
                
        #=============================================================================
        #=======================EMJESGSCMu========================================
        # variables for EMJESGSCMu j1.     
        E1_EMJESGSCMu = tree.j1_EMJESGSCMu_E
        Phi1_EMJESGSCMu = tree.j1_EMJESGSCMu_Phi
        Eta1_EMJESGSCMu = tree.j1_EMJESGSCMu_Eta
        Pt1_EMJESGSCMu = tree.j1_EMJESGSCMu_Pt
    
        # variables for EMJESGSCMu j2.     
        E2_EMJESGSCMu = tree.j2_EMJESGSCMu_E
        Phi2_EMJESGSCMu = tree.j2_EMJESGSCMu_Phi
        Eta2_EMJESGSCMu = tree.j2_EMJESGSCMu_Eta
        Pt2_EMJESGSCMu = tree.j2_EMJESGSCMu_Pt
    
        # PtRel in EMJESGSCMu
        Theta1_EMJESGSCMu = AuxiliarFunctions.theta(Eta1_EMJESGSCMu)
        Comp1_EMJESGSCMu = AuxiliarFunctions.Componets(E1_EMJESGSCMu,Eta1_EMJESGSCMu,Phi1_EMJESGSCMu,Theta1_EMJESGSCMu)
        norm1_EMJESGSCMu = AuxiliarFunctions.norm(Comp1_EMJESGSCMu)
    
        Theta2_EMJESGSCMu = AuxiliarFunctions.theta(Eta2_EMJESGSCMu)
        Comp2_EMJESGSCMu = AuxiliarFunctions.Componets(E2_EMJESGSCMu,Eta2_EMJESGSCMu,Phi2_EMJESGSCMu,Theta2_EMJESGSCMu)
        norm2_EMJESGCMu = AuxiliarFunctions.norm(Comp2_EMJESGSCMu)
    
        # sum of components.
        Comp3_EMJESGSCMu = AuxiliarFunctions.somaComponents(Comp1_EMJESGSCMu,Comp2_EMJESGSCMu)
        norm3_EMJESGSCMu = AuxiliarFunctions.norm(Comp3_EMJESGSCMu)
        numerador1_EMJESGSCMu = AuxiliarFunctions.numerador(Comp1_EMJESGSCMu,Comp3_EMJESGSCMu)    
        angle1_EMJESGSCMu = AuxiliarFunctions.alpha(numerador1_EMJESGSCMu,norm1_EMJESGSCMu,norm3_EMJESGSCMu)
    
        #compute Relative Pt
        PtRel_j1_EMJESGSCMu = AuxiliarFunctions.PtRelative(E1_EMJESGSCMu,angle1_EMJESGSCMu)
    
        #==========================
        #==========================
    
        # only for EM and j1.
        j1_FracEM3 = tree.j1_FracEM3
        j1_FracTile0 = tree.j1_FracTile0
        j1_TrkWidth = tree.j1_TrkWidth
        j1_EMF = tree.j1_EMF
        j1_JVF = tree.j1_JVF
        j1_NTrk = tree.j1_NTrk 
        j1_SumPtTrk = tree.j1_SumPtTrk
        # only for EM and j2.
        j2_FracEM3 = tree.j2_FracEM3
        j2_FracTile0 = tree.j2_FracTile0
        j2_TrkWidth = tree.j2_TrkWidth
        j2_EMF = tree.j2_EMF
        j2_JVF = tree.j2_JVF
        j2_NTrk = tree.j2_NTrk 
        j2_SumPtTrk = tree.j2_SumPtTrk
    
        
        NNinput = np.array([j1_FracEM3,j1_FracTile0,j1_TrkWidth,j1_EMF,j1_JVF,j1_NTrk,j1_SumPtTrk,Pt1_EMJESGSCMu,Eta1_EMJESGSCMu, Phi1_EMJESGSCMu, E1_EMJESGSCMu, PtRel_j1_EMJESGSCMu,
                            j2_FracEM3,j2_FracTile0,j2_TrkWidth,j2_EMF,j2_JVF,j2_NTrk,j2_SumPtTrk,Pt2_EMJESGSCMu,Eta2_EMJESGSCMu, Phi2_EMJESGSCMu, E2_EMJESGSCMu])#, PtRel_j1_EMJESGSCMu])

        NNinput_noPtRel = np.array([j1_FracEM3,j1_FracTile0,j1_TrkWidth,j1_EMF,j1_JVF,j1_NTrk,j1_SumPtTrk,Pt1_EMJESGSCMu,Eta1_EMJESGSCMu, Phi1_EMJESGSCMu, E1_EMJESGSCMu,
                            j2_FracEM3,j2_FracTile0,j2_TrkWidth,j2_EMF,j2_JVF,j2_NTrk,j2_SumPtTrk,Pt2_EMJESGSCMu,Eta2_EMJESGSCMu, Phi2_EMJESGSCMu, E2_EMJESGSCMu])#, PtRel_j1_EMJESGSCMu])
        
        V_NNinput = np.reshape(NNinput, (1,23))
        V_NNinput_noPtRel = np.reshape(NNinput_noPtRel, (1,22))

        norm_NNinput_TTbar = classifiers['TTbar'][10]['Fold 0']['pre_proc'].transform(V_NNinput)
        norm_NNinput_Wbb = classifiers['Wbb'][10]['Fold 0']['pre_proc'].transform(V_NNinput)

        norm_NNinput_TTbar_noPtRel = classifiers_noPtRel['TTbar'][10]['Fold 0']['pre_proc'].transform(V_NNinput_noPtRel)
        norm_NNinput_Wbb_noPtRel = classifiers_noPtRel['Wbb'][10]['Fold 0']['pre_proc'].transform(V_NNinput_noPtRel)

        
        output_NN_TTbar = models['TTbar']['Fold 0']['init_1'].predict(norm_NNinput_TTbar)
        output_NN_Wbb = models['Wbb']['Fold 0']['init_7'].predict(norm_NNinput_Wbb)

        output_NN_TTbar_noPtRel = models_noPtRel['TTbar']['Fold 0']['init_1'].predict(norm_NNinput_TTbar_noPtRel)  
        output_NN_Wbb_noPtRel = models_noPtRel['Wbb']['Fold 0']['init_7'].predict(norm_NNinput_Wbb_noPtRel)


        j1_LV = ROOT.TLorentzVector()
        j2_LV = ROOT.TLorentzVector()
            
        j1_LV.SetPtEtaPhiE(Pt1_EMJESGSCMu,
                           Eta1_EMJESGSCMu, Phi1_EMJESGSCMu,E1_EMJESGSCMu)
        j2_LV.SetPtEtaPhiE(Pt2_EMJESGSCMu,
                           Eta2_EMJESGSCMu, Phi2_EMJESGSCMu,E2_EMJESGSCMu)
            
        dijet_LV = j1_LV + j2_LV
        Mass = dijet_LV.M()
            
        if process == 'lvbb125': 
            list_hist_M_TTbar_noNN[0].Fill(Mass,tree.eventWeight)
            list_hist_M_Wbb_noNN[0].Fill(Mass,tree.eventWeight)
        if process == 'TTbar':
            list_hist_M_TTbar_noNN[1].Fill(Mass,tree.eventWeight)
        if process == 'Wbb':
            list_hist_M_Wbb_noNN[1].Fill(Mass,tree.eventWeight)

        
        if output_NN_TTbar>0.4:
            
            j1_LV = ROOT.TLorentzVector()
            j2_LV = ROOT.TLorentzVector()
            
            j1_LV.SetPtEtaPhiE(Pt1_EMJESGSCMu,
                           Eta1_EMJESGSCMu, Phi1_EMJESGSCMu,E1_EMJESGSCMu)
            j2_LV.SetPtEtaPhiE(Pt2_EMJESGSCMu,
                           Eta2_EMJESGSCMu, Phi2_EMJESGSCMu,E2_EMJESGSCMu)
            
            dijet_LV = j1_LV + j2_LV
            Mass = dijet_LV.M()
            
            if process == 'lvbb125': 
                list_hist_M_TTbar[0].Fill(Mass,tree.eventWeight)
            if process == 'TTbar':
                list_hist_M_TTbar[1].Fill(Mass,tree.eventWeight)

        if output_NN_Wbb>0.4:

            j1_LV = ROOT.TLorentzVector()
            j2_LV = ROOT.TLorentzVector()
            
            j1_LV.SetPtEtaPhiE(Pt1_EMJESGSCMu,
                           Eta1_EMJESGSCMu, Phi1_EMJESGSCMu,E1_EMJESGSCMu)
            j2_LV.SetPtEtaPhiE(Pt2_EMJESGSCMu,
                           Eta2_EMJESGSCMu, Phi2_EMJESGSCMu,E2_EMJESGSCMu)
            
            dijet_LV = j1_LV + j2_LV
            Mass = dijet_LV.M()
            
            if process == 'lvbb125': 
                list_hist_M_Wbb[0].Fill(Mass,tree.eventWeight)
            if process == 'Wbb':
                list_hist_M_Wbb[1].Fill(Mass,tree.eventWeight)


        if output_NN_TTbar_noPtRel>0.4:
            
            j1_LV = ROOT.TLorentzVector()
            j2_LV = ROOT.TLorentzVector()
            
            j1_LV.SetPtEtaPhiE(Pt1_EMJESGSCMu,
                           Eta1_EMJESGSCMu, Phi1_EMJESGSCMu,E1_EMJESGSCMu)
            j2_LV.SetPtEtaPhiE(Pt2_EMJESGSCMu,
                           Eta2_EMJESGSCMu, Phi2_EMJESGSCMu,E2_EMJESGSCMu)
            
            dijet_LV = j1_LV + j2_LV
            Mass = dijet_LV.M()
            
            if process == 'lvbb125': 
                list_hist_M_TTbar_noPtRel[0].Fill(Mass,tree.eventWeight)
            if process == 'TTbar':
                list_hist_M_TTbar_noPtRel[1].Fill(Mass,tree.eventWeight)

        if output_NN_Wbb_noPtRel>0.4:

            j1_LV = ROOT.TLorentzVector()
            j2_LV = ROOT.TLorentzVector()
            
            j1_LV.SetPtEtaPhiE(Pt1_EMJESGSCMu,
                           Eta1_EMJESGSCMu, Phi1_EMJESGSCMu,E1_EMJESGSCMu)
            j2_LV.SetPtEtaPhiE(Pt2_EMJESGSCMu,
                           Eta2_EMJESGSCMu, Phi2_EMJESGSCMu,E2_EMJESGSCMu)
            
            dijet_LV = j1_LV + j2_LV
            Mass = dijet_LV.M()
            
            if process == 'lvbb125': 
                list_hist_M_Wbb_noPtRel[0].Fill(Mass,tree.eventWeight)
            if process == 'Wbb':
                list_hist_M_Wbb_noPtRel[1].Fill(Mass,tree.eventWeight)

a = ROOT.TCanvas("a")
b = ROOT.TCanvas("b")
c = ROOT.TCanvas("c")
d = ROOT.TCanvas("d")
e = ROOT.TCanvas("e")
f = ROOT.TCanvas("f")


clone_list_M_TTbar_noNN = []
for idx, hist in enumerate(list_hist_M_TTbar_noNN):
    newhist = hist.Clone()
    newhist.SetLineColor(list_color_TTbar[idx])
    clone_list_M_TTbar_noNN.append(newhist)
clone_list_M_Wbb_noNN = []
for idx, hist in enumerate(list_hist_M_Wbb_noNN):
    newhist = hist.Clone()
    newhist.SetLineColor(list_color_Wbb[idx])
    clone_list_M_Wbb_noNN.append(newhist)

clone_list_M_TTbar = []
for idx, hist in enumerate(list_hist_M_TTbar):
    newhist = hist.Clone()
    newhist.SetLineColor(list_color_TTbar[idx])
    clone_list_M_TTbar.append(newhist)
clone_list_M_Wbb = []
for idx, hist in enumerate(list_hist_M_Wbb):
    newhist = hist.Clone()
    newhist.SetLineColor(list_color_Wbb[idx])
    clone_list_M_Wbb.append(newhist)

clone_list_M_TTbar_noPtRel = []
for idx, hist in enumerate(list_hist_M_TTbar_noPtRel):
    newhist = hist.Clone()
    newhist.SetLineColor(list_color_TTbar[idx])
    clone_list_M_TTbar_noPtRel.append(newhist)
clone_list_M_Wbb_noPtRel = []
for idx, hist in enumerate(list_hist_M_Wbb_noPtRel):
    newhist = hist.Clone()
    newhist.SetLineColor(list_color_Wbb[idx])
    clone_list_M_Wbb_noPtRel.append(newhist)    

#HistogramFunctions.Draw1DHists(list_hist_M,output_path)
HistogramFunctions.DrawList1DHistInCanvas(clone_list_M_TTbar_noPtRel, "Mass Plot", " Invariant Mass ", "Events", a)        
a.Print(plots_path+'ttbar_mass.pdf')
HistogramFunctions.DrawList1DHistInCanvas(clone_list_M_Wbb_noPtRel, "Mass Plot", " Invariant Mass ", "Events", b)        
b.Print(plots_path+'wbb_mass.pdf')
HistogramFunctions.DrawList1DHistInCanvas(clone_list_M_TTbar, "Mass Plot", " Invariant Mass ", "Events", c)        
c.Print(plots_path+'ttbar_mass.pdf')
HistogramFunctions.DrawList1DHistInCanvas(clone_list_M_Wbb, "Mass Plot", " Invariant Mass ", "Events", d)        
d.Print(plots_path+'wbb_mass.pdf')
HistogramFunctions.DrawList1DHistInCanvas(clone_list_M_TTbar_noNN, "Mass Plot", " Invariant Mass ", "Events", e)        
e.Print(plots_path+'ttbar_mass_noNN.pdf')
HistogramFunctions.DrawList1DHistInCanvas(clone_list_M_Wbb_noNN, "Mass Plot", " Invariant Mass ", "Events", f)        
f.Print(plots_path+'wbb_mass_noNN.pdf')


# no PtRel

c = ROOT.TCanvas("c")
d = ROOT.TCanvas("d")

clone_list_M_TTbar_noPtRel = []
for idx, hist in enumerate(list_hist_M_TTbar_noPtRel):
    newhist = hist.Clone()
    newhist.SetLineColor(list_color_TTbar[idx])
    clone_list_M_TTbar_noPtRel.append(newhist)
clone_list_M_Wbb_noPtRel = []
for idx, hist in enumerate(list_hist_M_Wbb_noPtRel):
    newhist = hist.Clone()
    newhist.SetLineColor(list_color_Wbb[idx])
    clone_list_M_Wbb_noPtRel.append(newhist)
    
#HistogramFunctions.Draw1DHists(list_hist_M,output_path)
HistogramFunctions.DrawList1DHistInCanvas(clone_list_M_TTbar_noPtRel, "Mass Plot", " Invariant Mass ", "Events", c)        
c.Print(plots_path+'ttbar_mass_tight_noPtRel.pdf')
HistogramFunctions.DrawList1DHistInCanvas(clone_list_M_Wbb_noPtRel, "Mass Plot", " Invariant Mass ", "Events", d)        
d.Print(plots_path+'wbb_mass_tight_noPtRel.pdf')


l = ROOT.TCanvas("k")

ROOT.gStyle.SetOptStat(0)

hs = ROOT.THStack('hs',"")
leg = ROOT.TLegend(0.9,0.7,0.7,0.9)

hs.Add(list_hist_M_Wbb_noNN[1])
leg.AddEntry(list_hist_M_Wbb_noNN[1],"Wbb","f")
hs.Add(list_hist_M_Wbb_noNN[0])
leg.AddEntry(list_hist_M_Wbb_noNN[0],"WH125","f")

hs.Draw('hist')
hs.GetXaxis().SetTitle("m_{b#bar{b}} [GeV]")
hs.GetYaxis().SetTitle("Arbitrary Units")
leg.Draw()
l.Print(plots_path+'stack_plot_Wbb_noNN.pdf')

l = ROOT.TCanvas("k")

ROOT.gStyle.SetOptStat(0)

hs = ROOT.THStack('hs',"")
leg = ROOT.TLegend(0.9,0.7,0.7,0.9)

hs.Add(list_hist_M_TTbar_noNN[1])
leg.AddEntry(list_hist_M_TTbar_noNN[1],"TTbar","f")
hs.Add(list_hist_M_TTbar_noNN[0])
leg.AddEntry(list_hist_M_TTbar_noNN[0],"WH125","f")

hs.Draw('hist')
hs.GetXaxis().SetTitle("m_{b#bar{b}} [GeV]")
hs.GetYaxis().SetTitle("Arbitrary Units")
leg.Draw()
l.Print(plots_path+'stack_plot_TTbar_noNN.pdf')



l = ROOT.TCanvas("k")

ROOT.gStyle.SetOptStat(0)

hs = ROOT.THStack('hs',"")
leg = ROOT.TLegend(0.9,0.7,0.7,0.9)

hs.Add(list_hist_M_Wbb[1])
leg.AddEntry(list_hist_M_Wbb[1],"Wbb","f")
hs.Add(list_hist_M_Wbb[0])
leg.AddEntry(list_hist_M_Wbb[0],"WH125","f")

hs.Draw('hist')
hs.GetXaxis().SetTitle("m_{b#bar{b}} [GeV]")
hs.GetYaxis().SetTitle("Arbitrary Units")
leg.Draw()
l.Print(plots_path+'stack_plot_Wbb_tight.pdf')

l = ROOT.TCanvas("k")

ROOT.gStyle.SetOptStat(0)

hs = ROOT.THStack('hs',"")
leg = ROOT.TLegend(0.9,0.7,0.7,0.9)

hs.Add(list_hist_M_TTbar[1])
leg.AddEntry(list_hist_M_TTbar[1],"TTbar","f")
hs.Add(list_hist_M_TTbar[0])
leg.AddEntry(list_hist_M_TTbar[0],"WH125","f")

hs.Draw('hist')
hs.GetXaxis().SetTitle("m_{b#bar{b}} [GeV]")
hs.GetYaxis().SetTitle("Arbitrary Units")
leg.Draw()
l.Print(plots_path+'stack_plot_TTbar_tight.pdf')

# no PtRel
l = ROOT.TCanvas("k")

ROOT.gStyle.SetOptStat(0)

hs = ROOT.THStack('hs',"")
leg = ROOT.TLegend(0.9,0.7,0.7,0.9)

hs.Add(list_hist_M_Wbb_noPtRel[1])
leg.AddEntry(list_hist_M_Wbb_noPtRel[1],"Wbb","f")
hs.Add(list_hist_M_Wbb_noPtRel[0])
leg.AddEntry(list_hist_M_Wbb_noPtRel[0],"WH125","f")

hs.Draw('hist')
hs.GetXaxis().SetTitle("m_{b#bar{b}} [GeV]")
hs.GetYaxis().SetTitle("Arbitrary Units")
leg.Draw()
l.Print(plots_path+'stack_plot_Wbb_tight_noPtRel.pdf')

l = ROOT.TCanvas("k")

ROOT.gStyle.SetOptStat(0)

hs = ROOT.THStack('hs',"")
leg = ROOT.TLegend(0.9,0.7,0.7,0.9)

hs.Add(list_hist_M_TTbar_noPtRel[1])
leg.AddEntry(list_hist_M_TTbar_noPtRel[1],"TTbar","f")
hs.Add(list_hist_M_TTbar_noPtRel[0])
leg.AddEntry(list_hist_M_TTbar_noPtRel[0],"WH125","f")

hs.Draw('hist')
hs.GetXaxis().SetTitle("m_{b#bar{b}} [GeV]")
hs.GetYaxis().SetTitle("Arbitrary Units")
leg.Draw()
l.Print(plots_path+'stack_plot_TTbar_tight_noPtRel.pdf')