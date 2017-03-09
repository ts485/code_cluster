#!/usr/bin/python
#TS:2014/10/07

"""This code makes plots of mass functions and analytic bias.This is particularly to match tags for ejected and non-ejected halos"""

import numpy as np
import os
import sys
import struct

from collections import defaultdict
import scipy.spatial
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
import matplotlib.pyplot as plt

import pdb
import csv
sys.path.append('/home/fas/padmanabhan/ts485/research/MultiDark/code/')
import calc_Vcir as cv
sys.path.append('/home/fas/padmanabhan/ts485/scratch/HACC/Conv/code/')
import write_halo2bin_conv as ww



plt.rcParams.update({'text.fontsize':20,
                     'axes.labelsize': 23,#ts:for xlabel
                     'legend.fontsize': 13,#ts:for legend
                     'xtick.labelsize': 25,
                     'ytick.labelsize': 25,
                     'ytick.major.width':1,
                     'ytick.major.size':14,
                     'ytick.minor.size':7,
                     'xtick.major.size':14,
                     'xtick.minor.size':7,
                     'axes.linewidth':2,
                     'lines.linewidth':3})
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
from matplotlib.ticker import FuncFormatter

#--------------------

class Halos:
    def __init__(self,filename,Lbox=250.):
        self.filename = filename.split('/')[-1]
        dt = np.dtype([('haloID','i8'),('hostFlag','i8'),('Mvir','f4'),('Vmax','f4'),('Rvir','f4'),('Rs','f4'),('Xoff','f4'),('Macc','f4'),('Vacc','f4'),('Acc_Rate_Inst','f4'),('Acc_Rate_100Myr','f4'),('Acc_Rate_Tdyn','f4'),('a_form','f4'),('TbyU','f4'),('a_acc','f4'),('a_first_acc','f4'),('M_first_acc','f4'),('V_first_acc','f4'),('x','f4'),('y','f4'),('z','f4'),('vx','f4'),('vy','f4'),('vz','f4')])
        #dt = np.dtype([('haloID','i8'),('hostFlag','i8'),('Mvir','f4'),('Vmax','f4'),('Rvir','f4'),('Rs','f4'),('Xoff','f4'),('Macc','f4'),('Vacc','f4'),('Acc_Rate_Inst','f4'),('Acc_Rate_100Myr','f4'),('Acc_Rate_Tdyn','f4'),('a_form','f4'),('TbyU','f4'),('x','f4'),('y','f4'),('z','f4'),('vx','f4'),('vy','f4'),('vz','f4')])
        data = np.loadtxt(filename,skiprows=1,dtype=dt)
        self.Lbox = Lbox

        self.haloID = data['haloID']
		      
        self.Mvir = data['Mvir']
        self.Vmax = data['Vmax']
        self.Rvir = data['Rvir']/1000.
        self.coords = np.array((data['x'],data['y'],data['z'])).T
        self.velocity = np.array((data['vx'],data['vy'],data['vz'])).T
        self.hostFlag = data['hostFlag'].astype('int')
        self.Rs = data['Rs']/1000.
        self.concen = self.Rvir/self.Rs
        self.a_form = data['a_form']
        self.Xoff = data['Xoff']
        self.Macc = data['Macc']
        self.Vacc = data['Vacc']
        self.Acc_Rate_Inst = data['Acc_Rate_Inst']
        self.Acc_Rate_100Myr = data['Acc_Rate_100Myr']
	self.Acc_Rate_Tdyn = data['Acc_Rate_Tdyn']
        self.TbyU = data['TbyU']
        self.a_acc = data['a_acc']
        self.a_first_acc = data['a_first_acc']
        self.M_first_acc = data['M_first_acc']
        self.V_first_acc = data['V_first_acc']

    def select_subhalos(self,filename):
	dt = np.dtype([('haloID','i8'),('hostFlag','i8'),('Mvir','f4'),('Vmax','f4'),('Rvir','f4'),('Rs','f4'),('Xoff','f4'),('Macc','f4'),('Vacc','f4'),('Acc_Rate_Inst','f4'),('Acc_Rate_100Myr','f4'),('Acc_Rate_Tdyn','f4'),('a_form','f4'),('TbyU','f4'),('a_acc','f4'),('a_first_acc','f4'),('M_first_acc','f4'),('V_first_acc','f4'),('x','f4'),('y','f4'),('z','f4'),('vx','f4'),('vy','f4'),('vz','f4')])

        #dt = np.dtype([('haloID','i8'),('hostFlag','i8'),('Mvir','f4'),('Vmax','f4'),('Rvir','f4'),('Rs','f4'),('Xoff','f4'),('Macc','f4'),('Vacc','f4'),('Acc_Rate_Inst','f4'),('Acc_Rate_100Myr','f4'),('Acc_Rate_Tdyn','f4'),('a_form','f4'),('TbyU','f4'),('x','f4'),('y','f4'),('z','f4'),('vx','f4'),('vy','f4'),('vz','f4')])
        data = np.loadtxt(filename,skiprows=1,dtype=dt)
		      
        self.Mvir_sub = data['Mvir']
        self.Vmax_sub = data['Vmax']
        self.Rvir_sub = data['Rvir']/1000.
        self.coords_sub = np.array((data['x'],data['y'],data['z'])).T
        self.velocity_sub = np.array((data['vx'],data['vy'],data['vz'])).T
        self.hostFlag_sub = data['hostFlag'].astype('int')
	self.Rs_sub = data['Rs']/1000.

	print self.Rs_sub,np.min(self.Rvir_sub)
	self.concen_sub = self.Rvir_sub/self.Rs_sub
	self.a_form_sub = data['a_form']
	self.Xoff_sub = data['Xoff']
	self.Macc_sub = data['Macc']
	self.Vacc_sub = data['Vacc']
	self.Acc_Rate_Inst_sub = data['Acc_Rate_Inst']
	self.Acc_Rate_100Myr_sub = data['Acc_Rate_100Myr']
	self.Acc_Rate_Tdyn_sub = data['Acc_Rate_Tdyn']
        self.TbyU_sub = data['TbyU']
        self.a_acc_sub = data['a_acc']
        self.a_first_acc_sub = data['a_first_acc']
        self.M_first_acc_sub = data['M_first_acc']
        self.V_first_acc_sub = data['V_first_acc']

	test_subSub = np.in1d(self.hostFlag_sub,self.haloID)

	#ts: which is faster; having subhalo_tag or not?
	subhalo_tag = self.match_haloID2(np.unique(self.hostFlag_sub),self.haloID)

	dict_host = {}
	for i,j in enumerate(self.haloID[subhalo_tag]):
		dict_host[j] = i
	order = np.array([dict_host[i] for i in self.hostFlag_sub[test_subSub]])
		
	return test_subSub,subhalo_tag[order]

    def subhalo_catalog(self,host_haloIDs):
	dict_host = {}
	for i,haloID in enumerate(np.unique(host_haloIDs)):
		dict_host[haloID] = np.where(host_haloIDs==haloID)[0]
        return dict_host

    def get_ejectedHalos(self,filename,ejected=False):
        """Need a input file to identify whether a halo is ejected or not."""
        #ts: Read in files
        data_tag = np.loadtxt(filename,skiprows=4)
        #ts: Criteria for ejected/non-ejected halos
        if ejected:
            test = (data_tag[:,1]==1) & (data_tag[:,3]==0) #ts:ejected halos
        else:
            test = (data_tag[:,1]==0) & (data_tag[:,3]==0)
        
        self.ejectID = data_tag[test][:,0]
        return self.ejectID


    def match_haloID2(self,ejectID,haloID):
        """need halo ID which you want to match."""
        order_halo = np.argsort(haloID)
	haloID = haloID[order_halo]
		

        self.ejectID = np.sort(ejectID)

        ar_order = np.empty(np.shape(self.ejectID)[0])
        key_id = np.empty(np.shape(self.ejectID)[0])
        
        j = 0
        k = 0
        for i,one_id in enumerate(haloID):    
            while (self.ejectID[j] <= one_id):
                if self.ejectID[j]==one_id:
                    ar_order[k] = order_halo[i]
		    key_id[k] = one_id
                    k += 1
                if one_id != haloID[i]:
                    print "Stop using enumerate and change k to sample['id'][i]."
                j += 1
                if j == np.shape(self.ejectID)[0]:
                    print 'bye'
                    break


	    if j == np.shape(self.ejectID)[0]:
                print 'bye'
		break
		
        order = ar_order[:k].astype('int')
        return order


    def periodic(self,coords,Lbox=1.0):
        test = coords >= Lbox
        coords[test] = coords[test]-Lbox
        test = coords < 0.0
        coords[test] = Lbox+coords[test]
        return coords


    def shift_to_redshiftSpace(self,coords,velocity,z=0.3,Lbox=250.):
        Cosmology = [0.27,0.0,0.73]
        hubble = ww.HubbleParameter(z,cosmology=Cosmology)
        print 'vel is ',np.max(velocity[:,2]/hubble)
        coords [:,2] += velocity[:,2]/hubble
        coords = periodic(coords,Lbox=Lbox)
        return coords
		    		

    def write_halos(self,outputfilename,coords,mass,vmax,weight=None):
        if weight==None:
		weight = np.ones(np.shape(coords)[0])
	print weight[:5]
	message = '#x, y, z, weight, Mvir, Vmax'
	fmt = '%f %f %f %f %f %f'
        np.savetxt(outputfilename,np.array((coords[:,0],coords[:,1],coords[:,2],weight,np.log10(mass),vmax)).T)

    def write_subhalos(self,outputfilename,coords,mvir,macc,vmax,vacc,weight=None):
        if weight==None:
		weight = np.ones(np.shape(coords)[0])
	print weight[:5]
	message = '#x, y, z, weight,Mvir_sub, Macc_sub, Vmax_sub, Vacc_sub'
	fmt = '%f %f %f %f %f %f %f %f'
        np.savetxt(outputfilename,np.array((coords[:,0],coords[:,1],coords[:,2],weight,np.log10(mvir),np.log10(macc),vmax,vacc)).T,header=message,fmt=fmt)
    
	    

def write_subhalos(outputfilename,coords,mass_host,mass_sub,weight=None):
    if weight==None:
	    weight = np.ones(np.shape(coords)[0])
    print weight[:5]
    np.savetxt(outputfilename,np.array((coords[:,0],coords[:,1],coords[:,2],weight,np.log10(mass_host),np.log10(mass_sub))).T)

def plot_subhalo_massFunction():

    for i,mass in enumerate(np.array((13.6,13.8,14.0,14.2,14.4,14.6,14.8))):
        test = (halos.Mvir[order_host] > 10**mass) & (halos.Mvir[order_host] < 10**(mass+0.2))
	N_host = np.shape(np.unique(halos.haloID[order_host][test]))[0]
	num1,bins = np.histogram(np.log10(halos.Vmax_sub[order_sub]/halos.Vmax[order_host])[test],bins=30,range=[-3.,0.0])
	norm = bins[1]-bins[0]
	plt.plot((bins[:-1]+bins[1:])/2.,num1/(N_host*norm),label='['+str(mass)+','+str(mass+0.2)+']')
    plt.legend()
    plt.semilogy()
    plt.xlabel(r'${\rm log}_{10}[V_{\rm max,sub}/V_{\rm max,host}]$')
    plt.ylabel(r'${\rm log}[dN/d{\rm log}(v/V)]$')
    plt.tight_layout()
    plt.axis([-3,0,10**-2,10])
    plt.savefig('subhalo_vmaxFunction_Msub10.pdf')

def plot_subhalo_radialDist():
    test0 = (halos.Mvir[order_host]>10**14)&(halos.Mvir_sub[order_sub]>10**11.5)
    coords_host = halos.coords[order_host][test0]
    coords_sub = halos.coords_sub[order_sub][test0]
    Mvir = halos.Mvir[order_host][test0]
    Rvir = halos.Rvir[order_host][test0]

    for i,mass in enumerate(np.array((13.6,13.8,14.0,14.2,14.4,14.6,14.8))):
        test = (Mvir > 10**mass) & (Mvir < 10**(mass+0.2))
	
	plt.plot((bins[:-1]+bins[1:])/2.,num1/(N_host*norm),label='['+str(mass)+','+str(mass+0.2)+']')
    plt.legend()
    plt.semilogy()
    plt.xlabel(r'${\rm log}_{10}[V_{\rm max,sub}/V_{\rm max,host}]$')
    plt.ylabel(r'${\rm log}[dN/d{\rm log}(v/V)]$')
    plt.tight_layout()
    plt.axis([-3,0,10**-2,10])
    plt.savefig('subhalo_vmaxFunction_Msub10.pdf')

def find_linearParams(data_x,data_y):
    num = np.shape(data_x)[0]
    S_xx = np.inner(data_x,data_x)
    S_yy = np.inner(data_y,data_y)
    S_xy = np.inner(data_x,data_y)
    S_x = np.sum(data_x)
    S_y = np.sum(data_y)
    Delta = num*S_xx-(S_x)**2.
    Delta2 = num*S_yy-(S_y)**2.
    #a = (S_xx*S_y-S_x*S_xy)/Delta
    b = (num*S_xy-S_x*S_y)/Delta
    a = np.median(data_y)-b*np.median(data_x)
    print 'coeficcient is ',b*((num*S_xy-S_x*S_y)/Delta2),b
    return a,b

        
def main(argv=None):  
    Lbox = 250.
    halos = Halos('../code/Bolshoi_hosthalos_Mvir13_z0.dat',Lbox=Lbox)

    order_sub,order_host = halos.select_subhalos('../code/Bolshoi_Mvir13_subhalos_z0.dat')


    test0 = (halos.Mvir[order_host] >10**13.0) & (halos.Mvir[order_host] <10**13.5)
    #Mvir = halos.Mvir[order_host][test0]
    #Mvir_sub = halos.Mvir_sub[order_sub][test0]
    #Macc_sub = halos.Macc_sub[order_sub][test0]
    #Rhost = halos.Rvir[order_host][test0]
    coords_host = halos.coords[order_host][test0]
    coords_sub = halos.coords_sub[order_sub][test0]
    vel_host = halos.velocity[order_host][test0]
    vel_sub = halos.velocity_sub[order_sub][test0]
    
    z_acc_first = 1./halos.a_first_acc_sub[order_sub][test0]-1        
    z_acc_sub = 1./halos.a_acc_sub[order_sub][test0]-1
    test1 = (z_acc_sub < 0.1)
    test2 = (z_acc_sub > 0.1) & (z_acc_sub < 0.25)
    test3 = (z_acc_sub > 0.25) & (z_acc_sub < 0.5)
    test4 = z_acc_sub>0.5
    velDiff = np.sqrt((vel_sub[:,0]-vel_host[:,0])**2.+(vel_sub[:,1]-vel_host[:,1])**2.+(vel_sub[:,2]-vel_host[:,2])**2.)
    hist1,bins = np.histogram(velDiff[test1],bins=100,range=[0,2000])
    hist2,bins = np.histogram(velDiff[test2],bins=100,range=[0,2000])
    hist3,bins = np.histogram(velDiff[test3],bins=100,range=[0,2000])
    hist4,bins = np.histogram(velDiff[test4],bins=100,range=[0,2000])
    np.savetxt('hist_velDiff_zacc_M13to13.5.dat',np.array(((bins[1:]+bins[:-1])/2.,hist1,hist2,hist3,hist4)).T)
    

    
    test1 = (z_acc_first < 0.1)
    test2 = (z_acc_first > 0.1) & (z_acc_first < 0.25)
    test3 = (z_acc_first > 0.25) & (z_acc_first < 0.5)
    test4 = z_acc_first>0.5
    hist1,bins = np.histogram(velDiff[test1],bins=100,range=[0,2000])
    hist2,bins = np.histogram(velDiff[test2],bins=100,range=[0,2000])
    hist3,bins = np.histogram(velDiff[test3],bins=100,range=[0,2000])
    hist4,bins = np.histogram(velDiff[test4],bins=100,range=[0,2000])
    np.savetxt('hist_velDiff_zacc_first_M13to13.5.dat',np.array(((bins[1:]+bins[:-1])/2.,hist1,hist2,hist3,hist4)).T)
    
        
    

if __name__ == "__main__":
    sys.exit(main())
