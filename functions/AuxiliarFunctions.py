import ROOT
import math
import numpy 
from numpy import linalg

# using to compute the relative pt

def theta(Eta):    
    theta = 2*(math.atan(pow(math.e,-Eta)))
    return theta

def Componets(E,Eta,Phi,theta):
    Componets = []
        
    px = ((E)*(math.sin(theta))*(math.cos(Phi)))
    Componets.append(px)
    
    py = ((E)*(math.sin(theta))*(math.sin(Phi)))
    Componets.append(py)
    
    pz = ((E)*(math.cos(theta)))
    Componets.append(pz)
 
    return Componets

def somaComponents(Comp1,Comp2):
    s_comp = []
    
    s_px  = Comp1[0]+Comp2[0]
    s_comp.append(s_px)
    
    s_py = Comp1[1]+Comp2[1]
    s_comp.append(s_py)
    
    s_pz = Comp1[2]+Comp2[2]
    s_comp.append(s_pz)
    
    return s_comp

def numerador(Comp,Comp3):
    numerador = ((Comp[0])*(Comp3[0])+(Comp[1])*(Comp3[1])+(Comp[2])*(Comp3[2])) 
    return numerador


def norm(Comp):
    px = Comp[0]
    py = Comp[1]
    pz = Comp[2]
    Vector = numpy.array([px,py,pz])
    norm = numpy.linalg.norm(Vector)
    return norm

def alpha(numerador,norm,norm3):
    alpha = math.acos(numerador/(norm*norm3))
    return alpha


def PtRelative(E,angle):
    PtRelative = E * (math.sin(angle))
    return PtRelative

# using to compute a difference between the j1 and j2 phi variables

def modulo(a,b):
    g1=math.degrees(a)
    if g1<0:
        g1 = 360 + g1
      
    g2=math.degrees(b)
    if g2<0:
        g2 = 360 + g2
     
    m1= g1-g2
    if m1<0:
        m1 = 360 + m1
    if m1>180:
        m1 = 360 - m1
       
    return round(m1,3)

