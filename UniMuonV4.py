#!/usr/bin/env python3

import sys

# Se verifica los parametros
if len(sys.argv) != 2:
    print("Usage: python UniMuonV2.py  Setting.txt")
    sys.exit(1)
#Se leen todos los parametros que trae el Setting para poder usarse
file = sys.argv[1]    
archivo = open(file, 'r')
contenido = archivo.read()
archivo.close()
palabras=contenido.splitlines()
run=0
n=0
h=0
p_i=0
p_f=0
z_i=0
z_f=0
a_i=0
a_f=0
seed=0
model=0
rout=""
for p in palabras:
    if(p.split(":")[0]=="RUN"):
        run=int(p.split(":")[1])
    elif(p.split(":")[0]=="NUMEROS DE MUONES"):
        n=int(p.split(":")[1])
    elif(p.split(":")[0]=="ALTURA"):
        h=int(p.split(":")[1])
    elif(p.split(":")[0]=="MOMENTO INICIAL"):
        p_i=int(p.split(":")[1])
    elif(p.split(":")[0]=="MOMENTO FINAL"):
        p_f=int(p.split(":")[1])
    elif(p.split(":")[0]=="ZENITAL INCIAL"):
        z_i=int(p.split(":")[1])
    elif(p.split(":")[0]=="ZENITAL FINAL"):
        z_f=int(p.split(":")[1])
    elif(p.split(":")[0]=="AZIMUTAL INICIAL"):
        a_i=int(p.split(":")[1])
    elif(p.split(":")[0]=="AZIMUTAL FINAL"):
        a_f=int(p.split(":")[1])
    elif(p.split(":")[0]=="MODELO"):
        model=(p.split(":")[1])    
    elif(p.split(":")[0]=="ARGUMENTOS"):
        arg=str(p.split(":")[1])    
    elif(p.split(":")[0]=="SEED"):
        seed=int(p.split(":")[1])
    elif(p.split(":")[0]=="RUTA SALIDA"):
        rout=(p.split(":")[1])
    else:
        print("Codigo no reconocido"+ str(p.split(":")[0]))
        sys.exit(1)
    
    
#Se imprime el mensaje de bienvenida
print("###########################################")
print("###########################################")
print("### Bienvenidos a UniMuon Version 4.0  ###")
print("### Generador de muones atmosfericos   ###")
print("###  Utilizando una parametrizacion    ###")
print("###  Se generan muones con Monte Carlo  ###")
print("###########################################")
print("###########################################")

#Se importa las librerias necesarias
import numpy as np
from scipy.optimize import curve_fit

#Cargar los modelos iniciales del primer modelo
Models={"MS1":[0.1258,0.0588,2.65,100,650,0],
        "MS2":[0.14,0.054,2.7,115/1.1,850/1.1,0],
        "MS3":[0.175,0.037,2.72,103,810,0],
        "MS4":[0.2576,0.054,2.77,115/1.1,850/1.1,0],
        "MS5":[0.26,0.054,2.78,115/1.1,850/1.1,0]}


def costheta(o):
    H=6370
    R=32
    return np.square((1-((1-(np.cos(o))**2)/((1+(H/R))**2))))

def Ec(E,o):
    return (E+ 0.00206*((1030/costheta(o))-120))

def Graisser(E,o,h,model):
    A,B,g,E2,E3,r=Models[model]
    E=Ec(E,o)
    return(A*(E**(-g))*((1/(1+E*(costheta(o)/E2))+(B/(1+E*(costheta(o)/E3))+r)))*np.exp(-h/(4900+750*E)))



def Bugaev(E,h,*o):
    p=((E**2)-(0.10566)**2)**(0.5)
    y=np.log10(p)
    if(p>1 and p<930):
        return ((2.950e-3*p**(-((0.0252*y**3)-(0.263*y**2)+(1.2743*y)+0.3061)))*np.exp(-h/(4900+750*p)))
    elif(p<1590):
        return ((1.781e-2*p**(-((0.304*y)+1.791)))*np.exp(-h/(4900+750*p)))
    elif(p<4.2e5):
        return ((1.435e1*p**(-(3.672)))*np.exp(-h/(4900+750*p)))
    else:
        return(10e3*p**(-4)*np.exp(-h/(4900+750*p)))

def costhetaN(o):
    p1,p2,p3,p4,p5=0.102573, -0.068287, 0.958633,0.0407253,0.817285
    return (0.14*((((np.cos(o))**2)+(p1)**2+ p2*(np.cos(o))**(p3)+p4*(np.cos(o))**(p5))/(1+(p1)**2+p2+p4))**(1/2))


def Guan(E,o,h):
    c=((1/(1+(1.1*E*costhetaN(o))/(115)))+(0.054/(1+(1.1*E*costhetaN(o))/(850))))
    return (0.14*(E*(1+(3.64)/(E*(costhetaN(o))**(1.29))))**(-2.7)*c*np.exp(-h/(4900+750*E)))


def pdf_Ener(x,f,o,h,*arg):
    return f(x,o,h,*arg)

def pdf_Ang(x):
    return (np.cos(x))**2

    
def grado(a):
    return (a*np.pi/180)

#Se crean los valores para poder generar los angulos zenitales 
x1=np.linspace(grado(z_i),grado(z_f),2000)
y1=np.sum(pdf_Ang(x1))

def genZentith():
    Zen=np.random.choice(x1, p=(pdf_Ang(x1)/y1))
    return Zen

def GenMomentum(o,p_i,p_f,h,f,*arg):
    x=np.linspace(p_i,p_f,10000)
    y=np.sum(pdf_Ener(x,f,o,h,*arg))
    E=np.random.choice(x, p=pdf_Ener(x,f,o,h,*arg)/y)
    p=np.sqrt((E**2)-(0.1057**2))
    return p

def genAzimuth():
    return np.random.uniform(grado(a_i), grado(a_f))





    
#Se crea un muon que tiene un angulo azimutal, zenital, tiene momento y la carga    
class muon():

     def __init__(self,p_i,p_f,h,f,*arg):
        self.zenitalAngle=genZentith()
        self.azimutalAngle=genAzimuth()
        self.momentum=GenMomentum(self.zenitalAngle,p_i,p_f,h,f,*arg)
        self.charge=-1
        
#Se crean dos objetos sky con el cual se puede generar un numero de muones definidos 
class sky:
    
    def __init__(self,n,p_i,p_f,h,f,*arg):
        self.muons=[]
        self.pInital=p_i
        self.pFinal=p_f
        self.h=h
        for i in range(n):
            muon1=muon(p_i,p_f,h,f,*arg)
            self.muons.append(muon1)
            
    def disMomentum(self):
        momentuns=[]
        for i in self.muons:
            momentuns.append(i.momentum)
        return momentuns
    
    def disZenital(self):
        Zenitals=[]
        for i in self.muons:
            Zenitals.append(i.zenitalAngle)
        return Zenitals
    
    def disAzimutal(self):
        azimutals=[]
        for i in self.muons:
            azimutals.append(i.azimutalAngle)
        return azimutals
    
    def muonsMomentum(self,p_i,p_f):
        momen=[]
        for i in self.muons:
            if(i.momentum>p_i and i.momentum<p_f):
                momen.append(i)
        return momen               
        
    def addmuon(self,n,pi,pf,f2,*arg):
        for i in range(n):
            muon1=muon(pi,pf,self.h,f2,*arg)
            self.muons.append(muon1)

#Guardar los datos calculados            
def guardar():
    mon=skyMain.disMomentum()
    azi=skyMain.disAzimutal()
    zen=skyMain.disZenital()
    data=[mon,azi,zen]
    fmt = '%.3f'
    ruta=str(rout)+"muonesR"+str(run)+".txt"
    np.savetxt(ruta, data, fmt=fmt)



if __name__ == '__main__':

    if(seed>0):
        np.random.seed(seed)
    funcion_a_llamar = globals().get(model)

    if callable(funcion_a_llamar):
        if(arg!="None"):
            skyMain=sky(n,p_i,p_f,h,funcion_a_llamar,arg)
        else:
            skyMain=sky(n,p_i,p_f,h,funcion_a_llamar)
    else:
        print("La funcion no existe o no es callable.")

    
    
    print("Los muones se han creado con exito")
    guardar()
