import ROOT
import colorsys

def GetHistInfo(hist,return_pave):
    # this function with results export in hist
    
    return_pave.SetTextColor(hist.GetLineColor())
    return_pave.AddText(hist.GetTitle())
    return_pave.AddText("Mean: "+str(round(hist.GetMean(),3))+" +- "+str(round(hist.GetMeanError(),3)))
    return_pave.AddText("Var: "+str(round(hist.GetRMS(),3))+" +- "+str(round(hist.GetRMSError(),3)))
    return_pave.AddText("MOP: "+str(hist.GetBinCenter(hist.GetMaximumBin())))
    return_pave.AddText("Skewness: "+str(round(hist.GetSkewness(),3)))
    return return_pave



def GetFitInfo(function,return_pave):
    # this function with results export in hist
    return_pave.SetTextColor(function.GetLineColor());
    return_pave.AddText("Model: "+function.GetName().title());
    return_pave.AddText("Mean_{Model}: "+str(round(function.GetHistogram().GetMean(),3))+" +- "+str(round(function.GetHistogram().GetMeanError(),3)));
    return_pave.AddText("Var_{Model}: "+str(round(function.GetHistogram().GetRMS(),3))+" +- "+str(round(function.GetHistogram().GetRMSError(),3)));
    return_pave.AddText("MOP_{Model}: "+str(function.GetHistogram().GetBinCenter(function.GetHistogram().GetMaximumBin())))
    return_pave.AddText("Skewness_{Model}: "+str(round(function.GetHistogram().GetSkewness(),3)))
    return_pave.AddText("#chi^{2}: "+str(round(function.GetChisquare()))+", NDF: "+str(function.GetNDF()))
    return_pave.AddText("Prob: "+str(100*function.GetProb()))
    return return_pave



