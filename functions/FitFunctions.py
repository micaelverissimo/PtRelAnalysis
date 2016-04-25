import ROOT

from FunctionsPrototypes import *

def fitBukin(hist,min,max,fit_type):
    n_par_bukin = 6
    f=ROOT.TF1("bukin",Bukin(),min,max,n_par_bukin)
    
    # Default Parameters
    if hist.GetName().find('trut') !=-1:
        print hist.GetName()
        f.SetParameter(0,0.1) # norm
    else:
        f.SetParameter(0,1000) # norm
    
    f.SetParameter(1,120) # x0
    f.SetParameter(2,10) # sigma
    f.SetParameter(3,0.1) # xi
    f.SetParameter(4,0.1) # par[4]
    f.SetParameter(5,0.1) # par[5]
    
    hist.Fit("bukin",fit_type)
    f_fit = hist.GetFunction("bukin")
    
    f_fit.SetLineWidth(hist.GetLineWidth())
    f_fit.SetLineColor(hist.GetLineColor())
    return f_fit
