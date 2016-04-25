import ROOT
import FigureFunctions

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
