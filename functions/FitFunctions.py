import ROOT
import numpy as np
from FunctionsTypes import *

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


def fitGauss(histogram,min,max,fit_options):
    number_of_parameters_gauss = 3
    f=ROOT.TF1("gauss",Gauss(),min,max,number_of_parameters_gauss)
    
    # Default Parameters
    f.SetParameter(0,1.0)# norm
    f.SetParameter(1,0.0)
    f.SetParameter(2,1.0)
    
    histogram.Fit("gauss",fit_options)
    f_fit = histogram.GetFunction("gauss")
    
    f_fit.SetLineWidth(histogram.GetLineWidth())
    f_fit.SetLineColor(histogram.GetLineColor())
    return f_fit

def fitRayleigh(histogram,min,max,fit_options):
    number_of_parameters_rayleigh = 2
    f=ROOT.TF1("rayleigh",Rayleigh(),min,max,number_of_parameters_rayleigh)
    norm = histogram.GetBinCenter(histogram.GetMaximumBin())
    # Default Parameters
    f.SetParameter(1,norm) # normalization Factor
    f.SetParameter(0,1.0)# scale of standard raylaigh distribution 
     #   f.SetParameter(1,0)
     #   f.SetParameter(2,1)
    
    histogram.Fit("rayleigh",fit_options)
    f_fit = histogram.GetFunction("rayleigh")
    
    f_fit.SetLineWidth(histogram.GetLineWidth())
    f_fit.SetLineColor(histogram.GetLineColor())
    return f_fit


def fitUniform(histogram,min,max,fit_options):
    number_of_parameters_uniform = 4
    f=ROOT.TF1("uniform",Uniform(),min,max,number_of_parameters_uniform)
    norm =1.0/histogram.GetEntries()
    # Default Parameters
    f.SetParameter(0,-2.0)# scale of standard uniform distribution
    f.SetParameter(1,2.0)# scale of standard uniform distribution
    f.SetParameter(2,4.0)# scale of standard uniform distribution
    f.SetParameter(3,4.0)# Normalization Factor
 
    
    histogram.Fit("uniform",fit_options)
    f_fit = histogram.GetFunction("uniform")
    
    f_fit.SetLineWidth(histogram.GetLineWidth())
    f_fit.SetLineColor(histogram.GetLineColor())
    return f_fit

def fitLaplace(histogram,min,max,fit_options):
    number_of_parameters_laplace = 3
    f=ROOT.TF1("laplace",Laplace(),min,max,number_of_parameters_laplace)
    
    # Default Parameters
    f.SetParameter(0,0.0)# scale of standard Laplace distribution
    f.SetParameter(1,1.0)# scale of standard Laplace distribution
    f.SetParameter(2,1.0)# normalization Factor
    
    
    histogram.Fit("laplace",fit_options)
    f_fit = histogram.GetFunction("laplace")
    
    f_fit.SetLineWidth(histogram.GetLineWidth())
    f_fit.SetLineColor(histogram.GetLineColor())
    return f_fit

def fitChiSquared(histogram,min,max,fit_options):
    number_of_parameters_chi_squared = 3
    f=ROOT.TF1("chi squared",ChiSquared(),min,max,number_of_parameters_chi_squared)
    norm = histogram.GetBinCenter(histogram.GetMaximumBin())
    # Default Parameters
    f.SetParameter(0,4.0)# scale of standard Chi Squared distribution (for Work only)
    #f.SetParameter(1,0.0)# scale of standard Chi Squared distribution (for Work only)
    f.SetParameter(1,norm)# normalization Factor
    
    
    histogram.Fit("chi squared",fit_options)
    f_fit = histogram.GetFunction("chi squared")
    
    f_fit.SetLineWidth(histogram.GetLineWidth())
    f_fit.SetLineColor(histogram.GetLineColor())
    return f_fit
