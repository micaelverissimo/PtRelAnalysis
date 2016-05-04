import ROOT
from ROOT import TLatex,TPad,TList,TH1,TH1F,TH2F,TH1D,TH2D,TFile,TTree,TCanvas,TLegend,SetOwnership,gDirectory,TObject,gStyle,gROOT,TLorentzVector,TGraph,TMultiGraph,TColor,TAttMarker,TLine,TDatime,TGaxis,TF1,THStack,TAxis,TStyle,TPaveText,TAttFill
import FigureFunctions
import copy



class OneDimHistInfo:
    title = ''
    name = ''
    nbins = 0
    min = 9999
    max = -9999
    xlabel = ''
    ylabel = ''

    def __init__(self,ntitle, nname, nnbins, nmin, nmax, nxlabel, nylabel):
        self.title = ntitle
        self.name = nname
        self.nbins = nnbins
        self.min = nmin
        self.max = nmax
        self.xlabel = nxlabel
        self.ylabel = nylabel
    def __len__(self):
        return 1


def CreateListOf1DHistograms(list_1dinfos,list_colors):
    if len(list_1dinfos) != len(list_colors):
        print 'len(list_1dinfos): ', len(list_1dinfos), ', len(list_colors): ',len(list_colors)
        print 'CreateListOf1DHistograms: List of Hist has different length of List of Colors'
        return
    
    list_hists = []
            
    for (i,hist_info) in  enumerate(list_1dinfos):
        histo = ROOT.TH1F(hist_info.name,hist_info.title,hist_info.nbins,hist_info.min,hist_info.max)
        histo.GetXaxis().SetTitle(hist_info.xlabel)
        histo.GetYaxis().SetTitle(hist_info.ylabel)
        histo.SetLineWidth(2)
        histo.SetLineColor(list_colors[i])
        list_hists.append(histo)
    return list_hists

def Draw1DHists(list_hist,output_path,fit_function = "",fit_type = ""):
    c = ROOT.TCanvas("c_draw_hist","Analysis Canvas", 800,600)
    
    for hist in list_hist:
        hist.Draw()
        hist_pave_text = ROOT.TPaveText(0.6, 0.75, 0.9, 0.9, "NDC")
        hist_pave_text = FigureFunctions.GetHistInfo(hist,hist_pave_text)
        hist_pave_text.SetBorderSize(1)
        hist_pave_text.SetFillColor(ROOT.kNone)
        hist_pave_text.Draw()
        ROOT.SetOwnership(hist_pave_text,0)
        c.Print(output_path+'/'+hist.GetName()+'.pdf')
        if (fit_function != ""):
            function = Fit1DHist(hist,fit_function,fit_type)
            function.SetLineColor(hist.GetLineColor())
            function.SetLineWidth(2)
            function.Draw("same")
            
            function_pave_text = ROOT.TPaveText(0.6, 0.6, 0.9, 0.75, "NDC")
            function_pave_text = FigureFunctions.GetFitInfo(function,function_pave_text)
            function_pave_text.SetFillColor(ROOT.kNone)
            function_pave_text.SetBorderSize(1)
            
            function_pave_text.Draw()
            hist_pave_text.Draw()
            c.Print(output_path+'/'+hist.GetName()+'_fit_'+fit_function+'.pdf')
    
    return

def Legend(hist,canvas):
    canvas.cd()
    hist_pave_text = ROOT.TPaveText(0.6, 0.75, 0.9, 0.9, "NDC")
    hist_pave_text = FigureFunctions.GetHistInfo(hist,hist_pave_text)
    hist_pave_text.SetBorderSize(1)
    hist_pave_text.SetFillColor(ROOT.kNone)
    hist_pave_text.Draw()
    canvas.Update()
    ROOT.SetOwnership(hist_pave_text,0)
    return 

def Draw1DHistInCanvas(list_hist,hist_id,canvas,fit_function = "",fit_type = ""):
    
    c = canvas.cd()
    
    
    
    hist = list_hist[hist_id]
    hist.SetLineWidth(1)
    hist.Draw("same")
    #legend = Legend(hist,c)
    hist_pave_text = ROOT.TPaveText(0.6, 0.75, 0.9, 0.9, "NDC")
    hist_pave_text = FigureFunctions.GetHistInfo(hist,hist_pave_text)
    hist_pave_text.SetBorderSize(1)
    hist_pave_text.SetFillColor(ROOT.kNone)
    hist_pave_text.Draw()
    ROOT.SetOwnership(hist_pave_text,0)
    c.Update()
    
    if (fit_function != ""):
        function = Fit1DHist(hist,fit_function,fit_type)
        function.SetLineColor(hist.GetLineColor())
        function.SetLineWidth(2)
        function.Draw("same")
        
        function_pave_text = ROOT.TPaveText(0.6, 0.6, 0.9, 0.75, "NDC")
        function_pave_text = FigureFunctions.GetFitInfo(function,function_pave_text)
        function_pave_text.SetFillColor(ROOT.kNone)
        function_pave_text.SetBorderSize(1)
      
        function_pave_text.Draw()
        hist_pave_text.Draw()
    return 

# you need make a clone list to use this function for plot without lose your list
def DrawList1DHistInCanvas(clone_list, ntitle, nxlabel, nylabel, canvas):
    
    c = canvas.cd()
    print "canvas", c
    #detect max value
    max_hist = -9999
    id_hist = -1
    for (i,iaux) in enumerate(clone_list):
        if clone_list[i].GetBinContent(clone_list[i].GetMaximumBin()) > max_hist:
            max_hist = clone_list[i].GetBinContent(clone_list[i].GetMaximumBin())
            id_hist = i
    
    aux = clone_list[id_hist].GetTitle()
    xaux = clone_list[id_hist].GetXaxis().GetTitle()
    yaux = clone_list[id_hist].GetYaxis().GetTitle()
    clone_list[id_hist].SetTitle(ntitle)
    clone_list[id_hist].GetXaxis().SetTitle(nxlabel)
    clone_list[id_hist].GetYaxis().SetTitle(nxlabel)
    clone_list[id_hist].Draw("same")
    clone_list[id_hist].SetTitle(aux)    
    clone_list[id_hist].GetXaxis().SetTitle(xaux)
    clone_list[id_hist].GetYaxis().SetTitle(yaux)    
    
    # draw the histograms
    leg = ROOT.TLegend(0.5,0.75,0.9,0.9)
    leg.SetFillColor(ROOT.kNone)
    
    list_pave = []
    size_vertical = 0.15
    
    # change automatically the size of each Pavel 
    while (0.75 -size_vertical*(len(clone_list)/2) < 0.2):
        size_vertical -= 0.005
    
    for (i,iaux) in enumerate(clone_list):
        leg.AddEntry(clone_list[i],clone_list[i].GetTitle(),"l");
        
        horizontal_start = 0.0
        horizontal_end = 0.0
        
        vertical_start = 0.0
        vertical_end = 0.0
        
        if i%2 == 0:
            horizontal_start = 0.5 
            horizontal_end = 0.7
        else:
            horizontal_start = 0.7
            horizontal_end = 0.9
        
        vertical_start = 0.75 -size_vertical*(i/2 + 1)
        vertical_end = 0.75 -size_vertical*(i/2)
        
        list_pave.append(ROOT.TPaveText(horizontal_start, vertical_start, horizontal_end, vertical_end, "NDC"))
        
        list_pave[i] = FigureFunctions.GetHistInfo(clone_list[i],list_pave[i])
        aux = clone_list[i].GetTitle()
        xaux = clone_list[i].GetXaxis().GetTitle()
        yaux = clone_list[i].GetYaxis().GetTitle()
        clone_list[i].SetTitle(ntitle)
        clone_list[i].GetXaxis().SetTitle(nxlabel)
        clone_list[i].GetYaxis().SetTitle(nylabel)
        clone_list[i].SetFillStyle(0)
        clone_list[i].Draw("same")
        #list_hist[i].SetTitle(aux)    
        #list_hist[i].GetXaxis().SetTitle(xaux)
        #list_hist[i].GetYaxis().SetTitle(yaux)
    
       
    leg.Draw()
    for (i,iaux) in enumerate(list_pave):
        list_pave[i].SetBorderSize(1)
        list_pave[i].SetFillColor(ROOT.kNone)
        list_pave[i].Draw()
        ROOT.SetOwnership(list_pave[i],0)
    #canvas.Print("Analysis.pdf")
    
    return 

def Fit1DHist(hist, function_name, fit_type):
    if (function_name == "bukin"):
        function = FitFunctions.fitBukin(hist,40,220,fit_type)
    return function

def Fit1DListHist(list_hist, function_name, fit_type):
    list_functions = []
    if (function_name == "bukin"):
        for hist in list_hist:
            function = FitFunctions.fitBukin(hist,hist.GetMinimum(),hist.GetMaximum(),fit_type)
            list_functions.append(function)
    return list_functions
