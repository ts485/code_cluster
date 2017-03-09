#!/usr/bin/env python
#ts:2013/09/17
#Getting most parts from Antonio's code.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib import gridspec
plt.rcParams.update({'text.fontsize':10,
                     'axes.labelsize': 10,#ts:for xlabel
                     'legend.fontsize': 10,#ts:for legend
                     'xtick.labelsize': 10,
                     'ytick.labelsize': 10,
                     'ytick.major.width':1,
                     'ytick.major.size':14,
                     'ytick.minor.size':7,
                     'xtick.major.size':14,
                     'xtick.minor.size':7,
                     'axes.linewidth':2,
                     'lines.linewidth':3})
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
from matplotlib.ticker import FuncFormatter
import numpy as np
import math
import sys
import glob
from scipy import interpolate
from numpy import matrix

class Xi:
    def __init__(self,bs=20,Lbox=1000.,header='',cosmo_dir='cosmo0',type_dir='pre_s'):

        self.Lbox = Lbox
        self.header = header
        self.type_dir = type_dir
        self.cosmo_dir = cosmo_dir
        infile=file('../data/data_xi/'+cosmo_dir+'/'+header+'/'+type_dir+'/corr_000_DD.dat','r')
        bins1=np.array(infile.readline().rstrip().split(' '),dtype='float')
        bins2=np.array(infile.readline().rstrip().split(' '),dtype='float')
        self.bins1 = bins1
        self.bins2 = bins2

        self.nb = np.shape(self.bins1)[0]-1
        self.nu = np.shape(self.bins2)[0]-1
        self.bs = bs #ts:Averaging over some bin number.
    
        self.ss=0.5*(bins1[0:self.nb-self.bs+1:self.bs]+bins1[self.bs-1:self.nb+1:self.bs]) #s-axis,(nb-1)/2 elements -> change to spherical coords
        self.mu=0.5*(bins2[0:self.nu]+bins2[1:self.nu+1]) #mu axis,nu elements

    def count_pairs(self,filename,object_type='DD'):
        if object_type == 'RR':
            data = np.loadtxt('../data/hod1_s_000_RR.dat',skiprows=2, comments='#')
        else:
            data = np.loadtxt(filename+'_'+object_type+'.dat', skiprows=2, comments='#')
        count1 = np.array([np.mean(data[(self.bs*i):(self.bs*(i+1))],axis=0) for i in np.arange(round((self.nb)/self.bs,0))])
        return count1


    def count_pairs_recon(self,filename,object_type='DD'):
        data = np.loadtxt(filename+'_'+object_type+'.dat', skiprows=2, comments='#')
        count1 = np.array([np.mean(data[(self.bs*i):(self.bs*(i+1))],axis=0) for i in np.arange(round((self.nb)/self.bs,0))])
        return count1



    def calc_xi(self,header):
        if 'pre' in self.type_dir:
            self.RR = self.count_pairs(header,object_type='RR')
            self.DD = self.count_pairs(header,object_type='DD')
            self.DR = self.count_pairs(header,object_type='DR')
        if 'post' in self.type_dir:
            self.RR = self.count_pairs_recon(header,object_type='RR')
            self.DD = self.count_pairs_recon(header,object_type='DD')
            self.DR = self.count_pairs_recon(header,object_type='DR')
        xi = (self.DD-2*self.DR+self.RR)/self.RR

        mono = np.mean(xi,axis=1)
        norm = np.mean(((3.*self.mu**2.-1.)/2.)**2.)     
        quad = np.mean(xi*(3.*self.mu**2.-1)/2.,axis=1)/norm
        return mono,quad



    def calc_xi_periodic(self,filename):
        self.DD = self.count_pairs(filename.split('_DD')[0],object_type = 'DD')    


        RR = 2*(2.*np.pi*np.outer((self.bins1[1:self.nb+1]**3.-self.bins1[0:self.nb]**3.),(self.bins2[1:self.nu+1]-self.bins2[0:self.nu]))/(3.*self.Lbox**3.))
        self.RR = np.array([np.mean(RR[(self.bs*i):(self.bs*(i+1))],axis=0) for i in np.arange(round((self.nb)/self.bs,0))])

        xi = self.DD/self.RR -1.
        mono = np.mean(xi,axis=1)
        norm = np.mean((3.*self.mu**2.-1.)**2.)     
        quad = np.mean(xi*(3.*self.mu**2.-1),axis=1)/norm
        return self.ss,mono,quad

class Xis(Xi):
    def __init__(self,header='hod8',bs=20,Lbox=1000.,cosmo_dir='cosmo0',type_dir='pre_r'):

        if 'post' in type_dir:
            files = glob.glob('../data/data_xi/'+cosmo_dir+'/'+header+'/'+type_dir+'/*DD*.dat')
        else:
            files = glob.glob('../data/data_xi/'+cosmo_dir+'/'+header+'/'+type_dir+'/*DD*.dat')
        files.sort()
        self.header = header
        self.type_dir = type_dir
        self.cosmo_dir = cosmo_dir
        self.n_file = len(files)
        Xi.__init__(self,bs=bs,Lbox=Lbox,header=self.header,type_dir=self.type_dir,cosmo_dir = self.cosmo_dir)
        self.monos = np.empty((self.n_file,np.shape(self.ss)[0]))
        self.quads = np.empty((self.n_file,np.shape(self.ss)[0]))
        print 'there are ',len(files),' files'
        for i,file1 in enumerate(files):
            #filename = '../data/data_xi/'+cosmo_dir+'/'+header+'/'+type_dir+'/corr_'+file1.split('_')[4].split('.dat')[0]
            filename = file1.split('_DD')[0]
            mono,quad = self.calc_xi(filename)
            self.monos[i] = mono
            self.quads[i] = quad

   
    

    def calc_mean(self):
        self.mono_std = self.monos.std(axis=0)
        self.mono_mean = self.monos.mean(axis=0)

        self.quad_std = self.quads.std(axis=0)
        self.quad_mean = self.quads.mean(axis=0)

    def calc_sum(self,n_sub=4,seed=0):
        """Averaging some number of xis."""
        print self.n_file/n_sub
        np.random.seed(seed=seed)
        order = np.random.permutation(np.arange(self.n_file))
        #order = np.arange(self.n_file)
        new_monos = np.empty((self.n_file/n_sub,np.shape(self.ss)[0]))
        new_quads = np.empty((self.n_file/n_sub,np.shape(self.ss)[0]))
        for i in np.arange(self.n_file/n_sub):
            new_monos[i] = np.mean(self.monos[order][n_sub*i:n_sub*(i+1)],axis=0)
            new_quads[i] = np.mean(self.quads[order][n_sub*i:n_sub*(i+1)],axis=0)
        self.monos = new_monos
        self.quads = new_quads

            


    def save_mean(self,outputfile_mono,outputfile_quad):
        self.calc_mean()
        np.savetxt(outputfile_mono,np.array((self.ss,mono_mean,mono_std)).T)
        np.savetxt(outputfile_quad,np.array((self.ss,quad_mean,quad_std)).T)

    def calc_r_covarianceMatrix(self,start=20,end=36,header='r'):
        """only use monopole terms to compute a covariance matrix."""
        self.calc_mean()
        mean = self.mono_mean[start:end]
        std_matrix = np.zeros((self.n_file,np.shape(mean)[0],np.shape(mean)[0]))
   
        for i,mono in enumerate(self.monos):
            std_matrix[i] = np.outer(mono[start:end],mono[start:end])

        self.cov = np.mean(std_matrix,axis=0)-np.outer(mean,mean)
        np.savetxt(self.header+'_'+self.type_dir+'_cov_z0.15.dat',self.cov)
        return self.cov

    def calc_s_covarianceMatrix(self,start=0,end=10):
        """using both monopole and quadrupole terms."""
        self.calc_mean()
        mean = np.concatenate((self.mono_mean[start:end],self.quad_mean[start:end]),axis=0)
        std_matrix = np.zeros((self.n_file,np.shape(mean)[0],np.shape(mean)[0]))

        i = 0
        for mono,quad in zip(self.monos,self.quads):
            xi = np.concatenate((mono[start:end],quad[start:end]),axis=0)
            std_matrix[i] = np.outer(xi,xi)
            i += 1

        self.cov = np.mean(std_matrix,axis=0)-np.outer(mean,mean)
        #np.savetxt(self.header+'_'+self.type_dir+'_cov_z0.15.dat',self.cov)
        return self.cov

class Eisenstein_and_Hu:
    def __init__(self,z):
        self.z = z
        self.h = 0.71
        self.omega_m0 = 0.2648
        self.omega_b = 0.02258/self.h**2.
        self.omega_c = 0.220
        self.n = 0.963
        self.omega_lambda = 0.7352
        self.omega_rad = 0.0
        self.omega_mhh = self.omega_m0*(self.h**2.)
        #self.w = 0.0
        self.theta_cmb = 1.01037037 #ts:from the paper:COBE result
        self.z_eq = 2.50*10**4.*self.omega_mhh*self.theta_cmb**-4.
        self.k_eq = 0.0746*self.omega_mhh*self.theta_cmb**-2.
        self.recom_b1 = 0.313*(self.omega_mhh**-0.419)*(1+0.607*self.omega_mhh**0.674)
        self.recom_b2 = 0.238*self.omega_mhh**0.223

        y = self.omega_mhh
        #self.z_d = 1291*y**0.251/(1+0.659*y**0.828)*(1+self.recom_b1*y**self.recom_b2)
        self.z_d = 1291*(y**0.251)*(1+self.recom_b1*(self.omega_b*self.h**2.)**self.recom_b2)/(1+0.659*y**0.828)
        self.a1 = (46.9*y)**0.670*(1.+(32.1*y)**-0.532)
        self.a2 = (12.0*y)**0.424*(1.+(45.0*y)**-0.582)
        self.b1 = 0.944/(1.+(458*y)**-0.708)
        self.b2 = (0.395*y)**-0.0266
    
    
    def R(self,z):
        #ts: is it omega_m or omega_b?
        return 31.5*self.omega_mhh*self.theta_cmb**-4.*(1000/z)

    def sound_horizon(self):
        #unit [Mpc/h]--make sure!
        R_d = self.R(self.z_d)
        R_eq = self.R(self.z_eq)
        Hubble = 100 #[km/s/(Mpc/h)]
        c = 3*10**5. #speed of light [km/s]
        return (2./3.)/self.k_eq*np.sqrt(6./R_eq)*np.log((np.sqrt(1.+R_d)+np.sqrt(R_d+R_eq))/(1+np.sqrt(R_eq)))*c/Hubble

    def alpha_c(self):
        return self.a1**(-self.omega_b/self.omega_m0)*self.a2**(-(self.omega_b/omega_m0)**3.)

    def inverse_beta_c(self):
        return 1.+self.b1*((self.omega_c/self.omega_m0)**self.b2-1.)

    def q_cdm(self,k_hmpc):
        k = k_hmpc*self.h
        #return k/(13.41* self.k_eq)
        return k*self.theta_cmb**2./self.omega_mhh

    def f(self,k_hmpc):
        k = k_hmpc*self.h
        return 1/(1+(k*self.sound_horizon()/5.4)**4.)

    def C(self,k_hmpc,alpha_cc):
        return 14.2/alpha_cc+386/(1+69.9*self.q_cdm(k_hmpc)**2.)

    def T_0(self,k_hmpc,alpha_cc,beta_cc):
        return np.log(np.exp(1.0)+1.8*beta_cc*self.q_cdm(k_hmpc))/(np.log(np.exp(1.0)+1.8*beta_cc*self.q_cdm(k_hmpc))+self.C(k_hmpc,alpha_cc)*self.q_cdb(k_hmpc)**2.)

    def T_c(self,k_hmpc):
        k = k_hmpc*self.h
        return self.f(k)*self.T_0(k,1,1/self.inverse_beta_c())+(1-self.f(k))*self.T_0(k,self.alpha_c(),1/self.inverse_beta_c())

    def alpha_gamma(self):
        return 1-0.328*np.log(431.0*self.omega_mhh)*self.omega_b/self.omega_m0+0.38*np.log(22.3*self.omega_mhh)*(self.omega_b/self.omega_m0)**2.

    def sound_horizon_fit_mpc(self):
        
        return 44.5*np.log(9.83/self.omega_mhh)/np.sqrt(1+10.0*(self.omega_b*self.h**2.)**0.75)

    def gamma_eff(self,k_hmpc):
        """this expression has extra h in the numerator (compared to the original eqn in the paper) but this seems the correct expression and the paper has typo."""
        k = k_hmpc*self.h
        return self.omega_mhh*(self.alpha_gamma()+(1-self.alpha_gamma())/(1+(0.43*k*self.sound_horizon_fit_mpc())**4.))

    def q_eff(self,k_hmpc):
        return self.q_cdm(k_hmpc)*self.omega_mhh/self.gamma_eff(k_hmpc)

    def C_0(self,k_hmpc):
        return 14.2 + 731.0/(1+62.5*self.q_eff(k_hmpc))

    def L_0(self,k_hmpc):
        return np.log(2.0*np.exp(1.0)+1.8*self.q_eff(k_hmpc))
    
    def TT_0(self,k_hmpc):
        return self.L_0(k_hmpc)/(self.L_0(k_hmpc)+self.C_0(k_hmpc)*self.q_eff(k_hmpc)**2.)

    def P_i(self,k_hmpc):
        return k_hmpc**self.n

    def P_0(self,k_hmpc):
        return self.TT_0(k_hmpc)**2.*self.P_i(k_hmpc)

    def dsquare(self,k_hmpc):
        return k_hmpc**3.*self.P_0(k_hmpc)/2/np.pi**2.

    def window_fn(self,k_hmpc,R_hmpc):
        kr = k_hmpc*R_hmpc
        return 3/(kr**3.) * (np.sin(kr)-kr*np.cos(kr))

    def growth_factor(self,z):
        omega_m = self.omega_m0*(1.+z)**3./(self.omega_lambda+self.omega_rad*(1.+z)**2.+self.omega_m0*(1.+z)**3.)
        omega_l = self.omega_lambda/(self.omega_lambda+self.omega_rad*(1.+z)**2.+self.omega_m0*(1.+z)**3.)

        return (1.+z)**(-1)*5*omega_m/2. *(omega_m**(4/7.)-omega_l+(1+omega_m/2.)*(1+omega_l/70))**(-1.)



        

class PowerSpectrum:
    def __init__(self,linear_file,beta=0.0,z=0.0,sigma_s=0.,sigma_para=0.,sigma_per=0.):
        power_lin = np.loadtxt(linear_file)
        k_lin = power_lin[:,0]
        linear = power_lin[:,1]
        self.beta = beta
        self.sigma_s = sigma_s
        self.sigma_para = sigma_para
        self.sigma_per = sigma_per
        cosmo = Eisenstein_and_Hu(z=z) #ts: need to change from 0 to z
        test = k_lin < 0.005
        self.factor = np.mean(linear[test]/cosmo.P_0(k_lin[test]))
        
        self.k = 10.**np.linspace(np.log10(np.min(k_lin)),np.log10(np.max(k_lin)),10000)
        self.no_wiggle = self.factor*cosmo.P_0(self.k) 
        self.linear = np.interp(self.k,k_lin,linear)
        self.power_dw = lambda mu:((self.linear-self.no_wiggle)*np.exp(-(self.k**2.)*(mu*mu*self.sigma_para**2.+(1-mu*mu)*self.sigma_per**2.)/2.)+self.no_wiggle)*(1+beta*mu*mu)**2./(1+(self.k*mu*self.sigma_s)**2.)**2.
        self.mus = np.linspace(0.0,1.0,201)
        self.mu_mean = (self.mus[:-1]+self.mus[1:])/2.
        power = np.array([self.power_dw(mu) for mu in self.mu_mean]).T
        self.f_power = interpolate.interp2d(self.k,self.mu_mean,power.T)

    def set_epsilon(self,epsilon):
        new_mus = np.array([np.sqrt(((1+epsilon)**-4.*mu**2.)/((1+epsilon)**-4.*mu**2.+(1+epsilon)**(2.)*(1-mu**2.))) for mu in self.mu_mean])
        power = np.array([self.f_power(self.k*np.sqrt((1+epsilon)**-4.*mu**2.+(1+epsilon)**2.*(1-mu**2.)),mu1) for mu1,mu in zip(new_mus,self.mu_mean)]).T
        self.power_mono = np.sum(power*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(self.mus[1:]-self.mus[:-1])
        L2 = (3*self.mu_mean*self.mu_mean-1.)/2.
        self.power_quad = np.sum(power*L2*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(L2*L2*(self.mus[1:]-self.mus[:-1]))

    def set_epsilon_beta(self,epsilon,beta):
        if self.beta != 0.0:
            print 'beta is not set to zero!!!'
        new_mus = np.array([np.sqrt(((1+epsilon)**-4.*mu**2.)/((1+epsilon)**-4.*mu**2.+(1+epsilon)**(2.)*(1-mu**2.))) for mu in self.mu_mean])
        power = np.array([((1+beta*mu1*mu1)**2.)*self.f_power(self.k*np.sqrt((1+epsilon)**-4.*mu**2.+(1+epsilon)**2.*(1-mu**2.)),mu1) for mu1,mu in zip(new_mus,self.mu_mean)]).T
        self.power_mono = np.sum(power*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(self.mus[1:]-self.mus[:-1])
        L2 = (3*self.mu_mean*self.mu_mean-1.)/2.
        self.power_quad = np.sum(power*L2*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(L2*L2*(self.mus[1:]-self.mus[:-1]))
                         
    def calc_fourierTransform(self,r_range=[0.1,250.],r_bin=100):
        self.r = np.linspace(r_range[0],r_range[1],r_bin)
        #ts:By multiplying pk by gaussian with 0.1[Mpc/h]to smooth
        y = lambda r: (1./(2*np.pi**2.))*(self.k**2.)*self.power_mono*np.exp(-(0.1*self.k)**2./2.)*np.sin(self.k*r)/(self.k*r) 
        y_quad = lambda r: (1./(2*np.pi**2.))*(self.k**2.)*self.power_quad*np.exp(-(0.1*self.k)**2./2.)*(-1)*((3/(self.k*r)**2.-1.)*(np.sin((self.k*r))/(self.k*r))-3*np.cos((self.k*r))/(self.k*r)**2.)
        self.xi_mono = np.array([np.trapz(y(r),self.k) for r in self.r])
        self.xi_quad = np.array([np.trapz(y_quad(r),self.k) for r in self.r])
        return self.r,self.xi_mono,self.xi_quad
 

    def xi_fit(self,r,alpha,B,A0,A1,A2,b,a0,a1,a2):
        f_mono = interpolate.interp1d(self.r,self.xi_mono,kind='linear')
        f_quad = interpolate.interp1d(self.r,self.xi_quad,kind='linear')
        xi = np.concatenate((B*f_mono(alpha*r)+A0+A1/r+A2/r**2.,b*f_quad(alpha*r)+a0+a1/r+a2/r**2.),axis=0)
        return xi

    def calc_chiSquare(self,r_obs,xi_obs,cov,alpha,B,A0,A1,A2,b,a0,a1,a2):
        cov = matrix(cov)
        xi_fit = self.xi_fit(r_obs,alpha,B,A0,A1,A2,b,a0,a1,a2)
        vector = np.inner(xi_obs-xi_fit,cov.I)
        #return np.inner(vector,xi_obs-xi_fit)+(np.log(alpha))**2./0.15**2.
        return np.inner(vector,xi_obs-xi_fit)


    def find_linearParams(self,r_obs,xi_obs,cov_inv,alpha):
        f_mono = interpolate.interp1d(self.r,self.xi_mono,kind='linear')
        f_quad = interpolate.interp1d(self.r,self.xi_quad,kind='linear')
        num = np.shape(r_obs)[0]
        xi_nw0 = f_mono(alpha*r_obs)
        xi_nw2 = f_quad(alpha*r_obs)
        zeros = np.zeros((4,num))
        matA_mono = np.concatenate((np.array((xi_nw0,np.ones(num),1./r_obs,1/r_obs**2.)),zeros),axis=1)
        matA_quad = np.concatenate((zeros,np.array((xi_nw2,np.ones(num),1./r_obs,1/r_obs**2.))),axis=1)
        matA = np.concatenate((matA_mono,matA_quad),axis=0)
        matA2 = np.inner(matA,cov_inv)
        matA3 = matrix(np.inner(matA2,matA))
        beta=np.inner(matA2,xi_obs)
        params = np.inner(matA3.I,beta)
        return params
        
    def least_squareFit(self,r_obs,xi_obs,cov):
        cov = matrix(cov)
        cov_inv = cov.I

        num_a = 200
        num_e = 100
        alphas = np.linspace(0.9,1.1,num=num_a)
        epsilons = np.linspace(-0.02,0.02,num=num_e)
        chis = np.zeros((num_e,num_a))
        params = np.zeros((num_e,num_a,8))
        xi0 = np.zeros((num_e,100))
        xi2 = np.zeros((num_e,100))
                     
        
        for i,epsilon in enumerate(epsilons):
            self.set_epsilon(epsilon)
            r,xi0[i],xi2[i] = self.calc_fourierTransform(r_range=[0.1,250.],r_bin=100)
        
            for j,alpha in enumerate(alphas):
                params[i,j] = self.find_linearParams(r_obs,xi_obs,cov_inv,alpha)
                chis[i,j] = self.calc_chiSquare(r_obs,xi_obs,cov,alpha,params[i,j,0],params[i,j,1],params[i,j,2],params[i,j,3],params[i,j,4],params[i,j,5],params[i,j,6],params[i,j,7])
         
        return chis,alphas,epsilons,params,xi0,xi2

    def least_squareFit_beta(self,r_obs,xi_obs,cov):
        cov = matrix(cov)
        cov_inv = cov.I

        num_a = 20
        num_e = 50
        num_beta = 40
        alphas = np.linspace(0.9,1.1,num=num_a)
        epsilons = np.linspace(-0.05,0.05,num=num_e)
        betas = np.linspace(0.0,0.6,num=num_beta)
        chis = np.zeros((num_e,num_a,num_beta))
        params = np.zeros((num_e,num_a,num_beta,8))
                     
        xi0 = np.zeros((num_e,num_beta,100))
        xi2 = np.zeros((num_e,num_beta,100))
                     

        for k,beta in enumerate(betas):
            for i,epsilon in enumerate(epsilons):               
                self.set_epsilon_beta(epsilon,beta)                
                r,xi0[i,k],xi2[i,k] = self.calc_fourierTransform(r_range=[0.1,250.],r_bin=100)
                for j,alpha in enumerate(alphas):
                    params[i,j,k] = self.find_linearParams(r_obs,xi_obs,cov_inv,alpha)
                    chis[i,j,k] = self.calc_chiSquare(r_obs,xi_obs,cov,alpha,params[i,j,k,0],params[i,j,k,1],params[i,j,k,2],params[i,j,k,3],params[i,j,k,4],params[i,j,k,5],params[i,j,k,6],params[i,j,k,7])
         
        return chis,alphas,epsilons,betas,params,xi0,xi2


class Decompose_beta(PowerSpectrum):
    def __init__(self,linear_file,z=0.0,sigma_s=0.,sigma_para=0.,sigma_per=0.):
        PowerSpectrum.__init__(self,linear_file,beta=0.0,z=z,sigma_s=sigma_s,sigma_para=sigma_para,sigma_per=sigma_per)


    def calc_fourierTransform(self,x,y_func):
        y = lambda r: (1./(2*np.pi**2.))*(self.k**2.)*y_func*np.exp(-(0.1*self.k)**2./2.)*np.sin(self.k*r)/(self.k*r) 
        return np.array([np.trapz(y(r),self.k) for r in x])

    def calc_fourierTransform_2(self,x,y_func):
        y = lambda r: (1./(2*np.pi**2.))*(self.k**2.)*y_func*np.exp(-(0.1*self.k)**2./2.)*(-1)*((3/(self.k*r)**2.-1.)*(np.sin((self.k*r))/(self.k*r))-3*np.cos((self.k*r))/(self.k*r)**2.)
        return np.array([np.trapz(y(r),self.k) for r in x])

    def calc_fourierTransform_4(self,x,y_func):
        y = lambda r: (1./(2*np.pi**2.))*(self.k**2.)*y_func*np.exp(-(0.1*self.k)**2./2.)*np.sin(self.k*r)/(self.k*r) 
        return np.array([np.trapz(y(r),self.k) for r in x])

    def calc_fourierTransform_6(self,x,y_func):
        y = lambda r: (1./(2*np.pi**2.))*(self.k**2.)*y_func*np.exp(-(0.1*self.k)**2./2.)*np.sin(self.k*r)/(self.k*r) 
        return np.array([np.trapz(y(r),self.k) for r in x])

    def calc_fourierTransform_8(self,x,y_func):
        y = lambda r: (1./(2*np.pi**2.))*(self.k**2.)*y_func*np.exp(-(0.1*self.k)**2./2.)*np.sin(self.k*r)/(self.k*r) 
        return np.array([np.trapz(y(r),self.k) for r in x])

    def calc_fourierTransform_10(self,x,y_func):
        y = lambda r: (1./(2*np.pi**2.))*(self.k**2.)*y_func*np.exp(-(0.1*self.k)**2./2.)*np.sin(self.k*r)/(self.k*r) 
        return np.array([np.trapz(y(r),self.k) for r in x])

    def L2(self,mu):
        return  (3*mu*mu-1.)/2.
    def L4(self,mu):
        return (35*mu**4.-30*mu**2.+3)/8.
    
    def L6(self,mu):
        return (231*mu**6.-315*mu**4.+105*mu**2.-5)/16.
    
    def L8(self,mu):
        return (6435*mu**8.-12012*mu**6.+6930*mu**4.-1260*mu**2.+35)/128.

    def L10(self,mu):
        return (46189*mu**10-109395*mu**8.+90090*mu**6.-30030*mu**4.+3465*mu**2.-63)/256.
    

    def set_templateFn(self):
        self.r = np.linspace(0.1,300.,300)
        power1 = np.array([self.power_dw(mu) for mu in self.mu_mean]).T        
        power1_mono = np.sum(power1*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(self.mus[1:]-self.mus[:-1])
        l2 = self.L2(self.mu_mean)
        l4 = self.L4(self.mu_mean)
        l6 = self.L6(self.mu_mean)
        l8 = self.L8(self.mu_mean)
        l10 = self.L10(self.mu_mean)
        power1_quad = np.sum(power1*l2*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l2*l2*(self.mus[1:]-self.mus[:-1]))
        power1_4 = np.sum(power1*l4*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l4*l4*(self.mus[1:]-self.mus[:-1]))
        power1_6 = np.sum(power1*l6*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l6*l6*(self.mus[1:]-self.mus[:-1]))
        power1_8 = np.sum(power1*l8*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l8*l8*(self.mus[1:]-self.mus[:-1]))
        power1_10 = np.sum(power1*l10*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l10*l10*(self.mus[1:]-self.mus[:-1]))


        power2 = np.array([mu*mu*self.power_dw(mu) for mu in self.mu_mean]).T        
        power2_mono = np.sum(power2*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(self.mus[1:]-self.mus[:-1])
        power2_quad = np.sum(power2*l2*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l2*l2*(self.mus[1:]-self.mus[:-1]))
        power2_4 = np.sum(power2*l4*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l4*l4*(self.mus[1:]-self.mus[:-1]))
        power2_6 = np.sum(power2*l6*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l6*l6*(self.mus[1:]-self.mus[:-1]))
        power2_8 = np.sum(power2*l8*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l8*l8*(self.mus[1:]-self.mus[:-1]))
        power2_10 = np.sum(power2*l10*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l10*l10*(self.mus[1:]-self.mus[:-1]))


        power3 = np.array([(mu**4.)*self.power_dw(mu) for mu in self.mu_mean]).T        
        power3_mono = np.sum(power3*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(self.mus[1:]-self.mus[:-1])
        power3_quad = np.sum(power3*l2*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l2*l2*(self.mus[1:]-self.mus[:-1]))
        power3_4 = np.sum(power3*l4*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l4*l4*(self.mus[1:]-self.mus[:-1]))
        power3_6 = np.sum(power3*l6*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l6*l6*(self.mus[1:]-self.mus[:-1]))
        power3_8 = np.sum(power3*l8*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l8*l8*(self.mus[1:]-self.mus[:-1]))
        power3_10 = np.sum(power3*l10*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l10*l10*(self.mus[1:]-self.mus[:-1]))


        self.xi1_0 = self.calc_fourierTransform(self.r,power1_mono)
        self.xi1_2 = self.calc_fourierTransform(self.r,power1_quad)
        self.xi1_4 = self.calc_fourierTransform(self.r,power1_4)
        self.xi1_6 = self.calc_fourierTransform(self.r,power1_6)
        self.xi1_8 = self.calc_fourierTransform(self.r,power1_8)
        self.xi1_10 = self.calc_fourierTransform(self.r,power1_10)
        self.xi2_0 = self.calc_fourierTransform(self.r,power2_mono)
        self.xi2_2 = self.calc_fourierTransform(self.r,power2_quad)
        self.xi2_4 = self.calc_fourierTransform(self.r,power2_4)
        self.xi2_6 = self.calc_fourierTransform(self.r,power2_6)
        self.xi2_8 = self.calc_fourierTransform(self.r,power2_8)
        self.xi2_10 = self.calc_fourierTransform(self.r,power2_10)
        self.xi3_0 = self.calc_fourierTransform(self.r,power3_mono)
        self.xi3_2 = self.calc_fourierTransform(self.r,power3_quad)
        self.xi3_4 = self.calc_fourierTransform(self.r,power3_4)
        self.xi3_6 = self.calc_fourierTransform(self.r,power3_6)
        self.xi3_8 = self.calc_fourierTransform(self.r,power3_8)
        self.xi3_10 = self.calc_fourierTransform(self.r,power3_10)


    def set_interpolation(self):
        self.f1_xi0 = interpolate.interp1d(self.r,self.xi1_0,kind='linear')
        self.f1_xi2 = interpolate.interp1d(self.r,self.xi1_2,kind='linear')
        self.f1_xi4 = interpolate.interp1d(self.r,self.xi1_4,kind='linear')
        self.f1_xi6 = interpolate.interp1d(self.r,self.xi1_6,kind='linear')
        self.f1_xi8 = interpolate.interp1d(self.r,self.xi1_8,kind='linear')
        self.f1_xi10 = interpolate.interp1d(self.r,self.xi1_10,kind='linear')
        self.f2_xi0 = interpolate.interp1d(self.r,self.xi2_0,kind='linear')
        self.f2_xi2 = interpolate.interp1d(self.r,self.xi2_2,kind='linear')
        self.f2_xi4 = interpolate.interp1d(self.r,self.xi2_4,kind='linear')
        self.f2_xi6 = interpolate.interp1d(self.r,self.xi2_6,kind='linear')
        self.f2_xi8 = interpolate.interp1d(self.r,self.xi2_8,kind='linear')
        self.f2_xi10 = interpolate.interp1d(self.r,self.xi2_10,kind='linear')
        self.f3_xi0 = interpolate.interp1d(self.r,self.xi3_0,kind='linear')
        self.f3_xi2 = interpolate.interp1d(self.r,self.xi3_2,kind='linear')
        self.f3_xi4 = interpolate.interp1d(self.r,self.xi3_4,kind='linear')
        self.f3_xi6 = interpolate.interp1d(self.r,self.xi3_6,kind='linear')
        self.f3_xi8 = interpolate.interp1d(self.r,self.xi3_8,kind='linear')
        self.f3_xi10 = interpolate.interp1d(self.r,self.xi3_10,kind='linear')
        
    def set_epsilon_beta(self,epsilon,beta):
        #ts: change to exact expression
        self.r2 = np.linspace(0.5,250,100)
        new_rs = np.array([self.r2*np.sqrt((1+epsilon)**4.*mu**2.+(1+epsilon)**(-2.)*(1-mu**2.)) for mu in self.mu_mean])
        new_mus = np.array([np.sqrt(((1+epsilon)**4.*mu**2.)/((1+epsilon)**4.*mu**2.+(1+epsilon)**(-2.)*(1-mu**2.))) for mu in self.mu_mean])
        xi0 = np.array([self.f1_xi0(new_r)+2*beta*self.f2_xi0(new_r)+beta**2.*self.f3_xi0(new_r) for new_r,new_mu in zip(new_rs,new_mus)])

        xi2 = np.array([self.L2(mu*np.sqrt(1.+6*epsilon*(1-mu*mu)))*(self.f1_xi2(new_r)+2*beta*self.f2_xi2(new_r)+beta**2.*self.f3_xi2(new_r)) for new_r,new_mu in zip(new_rs,new_mus)])

        xi4 = np.array([self.L4(mu*np.sqrt(1.+6*epsilon*(1-mu*mu)))*(self.f1_xi4(new_r)+2*beta*self.f2_xi4(new_r)+beta**2.*self.f3_xi4(new_r)) for new_r,new_mu in zip(new_rs,new_mus)])

        xi6 = np.array([self.L6(mu*np.sqrt(1.+6*epsilon*(1-mu*mu)))*(self.f1_xi6(new_r)+2*beta*self.f2_xi6(new_r)+beta**2.*self.f3_xi6(new_r)) for new_r,new_mu in zip(new_rs,new_mus)])

        xi8 = np.array([self.L8(mu*np.sqrt(1.+6*epsilon*(1-mu*mu)))*(self.f1_xi8(new_r)+2*beta*self.f2_xi8(new_r)+beta**2.*self.f3_xi8(new_r)) for new_r,new_mu in zip(new_rs,new_mus)])

        xi10 = np.array([self.L10(mu*np.sqrt(1.+6*epsilon*(1-mu*mu)))*(self.f1_xi10(new_r)+2*beta*self.f2_xi10(new_r)+beta**2.*self.f3_xi10(new_r)) for new_r,new_mu in zip(new_rs,new_mus)])
        xi = (xi0+xi2+xi4+xi6+xi8+xi10).T

        self.xi_mono = np.sum(xi*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum((self.mus[1:]-self.mus[:-1]))
        l2 = self.L2(self.mu_mean)
        self.xi_quad = np.sum(xi*l2*(self.mus[1:]-self.mus[:-1]),axis=1)/np.sum(l2*l2*(self.mus[1:]-self.mus[:-1]))
        return self.xi_mono,self.xi_quad


    def xi_fit(self,r,alpha,B,A0,A1,A2,b,a0,a1,a2):
        f_mono = interpolate.interp1d(self.r2,self.xi_mono,kind='linear')
        f_quad = interpolate.interp1d(self.r2,self.xi_quad,kind='linear')
        xi = np.concatenate((B*f_mono(alpha*r)+A0+A1/r+A2/r**2.,b*f_quad(alpha*r)+a0+a1/r+a2/r**2.),axis=0)
        return xi

    def calc_chiSquare(self,r_obs,xi_obs,cov,alpha,B,A0,A1,A2,b,a0,a1,a2):
        cov = matrix(cov)
        xi_fit = self.xi_fit(r_obs,alpha,B,A0,A1,A2,b,a0,a1,a2)
        vector = np.inner(xi_obs-xi_fit,cov.I)
        #return np.inner(vector,xi_obs-xi_fit)+(np.log(alpha))**2./0.15**2.
        return np.inner(vector,xi_obs-xi_fit)


    def find_linearParams(self,r_obs,xi_obs,cov_inv,alpha):
        f_mono = interpolate.interp1d(self.r2,self.xi_mono,kind='linear')
        f_quad = interpolate.interp1d(self.r2,self.xi_quad,kind='linear')
        num = np.shape(r_obs)[0]
        xi_nw0 = f_mono(alpha*r_obs)
        xi_nw2 = f_quad(alpha*r_obs)
        zeros = np.zeros((4,num))
        matA_mono = np.concatenate((np.array((xi_nw0,np.ones(num),1./r_obs,1/r_obs**2.)),zeros),axis=1)
        matA_quad = np.concatenate((zeros,np.array((xi_nw2,np.ones(num),1./r_obs,1/r_obs**2.))),axis=1)
        matA = np.concatenate((matA_mono,matA_quad),axis=0)
        matA2 = np.inner(matA,cov_inv)
        matA3 = matrix(np.inner(matA2,matA))
        beta=np.inner(matA2,xi_obs)
        params = np.inner(matA3.I,beta)
        return params
        

    def least_squareFit_beta(self,r_obs,xi_obs,cov):
        cov = matrix(cov)
        cov_inv = cov.I

        num_a = 20
        num_e = 50
        num_beta = 9
        alphas = np.linspace(0.9,1.1,num=num_a)
        epsilons = np.linspace(-0.05,0.05,num=num_e)
        betas = np.linspace(0.0,0.8,num=num_beta)
        chis = np.zeros((num_e,num_a,num_beta))
        params = np.zeros((num_e,num_a,num_beta,8))
                     
        xi0 = np.zeros((num_e,num_beta,100))
        xi2 = np.zeros((num_e,num_beta,100))
        
                     

        for k,beta in enumerate(betas):
            for i,epsilon in enumerate(epsilons):  
                xi0[i,k],xi2[i,k] = self.set_epsilon_beta(epsilon,beta)   
                
                for j,alpha in enumerate(alphas):
                    params[i,j,k] = self.find_linearParams(r_obs,xi_obs,cov_inv,alpha)
                    chis[i,j,k] = self.calc_chiSquare(r_obs,xi_obs,cov,alpha,params[i,j,k,0],params[i,j,k,1],params[i,j,k,2],params[i,j,k,3],params[i,j,k,4],params[i,j,k,5],params[i,j,k,6],params[i,j,k,7])
         
        return chis,alphas,epsilons,betas,params,xi0,xi2

def calc_fourierTransform(k,pk,r_range=[0.1,250.],r_bin=100,xi_type='mono'):
    k_new = np.linspace(np.min(k),np.max(k),20000)
    pk = np.interp(k_new,k,pk)
    r = np.linspace(r_range[0],r_range[1],r_bin)
    #ts:By multiplying pk by gaussian with 0.1[Mpc/h]to smooth
    if xi_type=='mono':
        y = lambda r: (1./(2*np.pi**2.))*(k_new**2.)*pk*np.exp(-(0.1*k_new)**2./2.)*np.sin(k_new*r)/(k_new*r) 
    if xi_type=='quad':        
        y = lambda r: (1./(2*np.pi**2.))*(k_new**2.)*pk*np.exp(-(0.1*k_new)**2./2.)*(-1)*((3/(k_new*r)**2.-1.)*(np.sin((k_new*r))/(k_new*r))-3*np.cos((k_new*r))/(k_new*r)**2.)
    xi = np.array([np.trapz(y(r1),k_new) for r1 in r])
    return r,xi

    

def main(srgv=None):
    import calc_xi
    dist = [15,40]_s

    header = sys.argv[1]
    recon = 'pre'
    space = 'r'

    xis=Xis(bs=4,header=header,cosmo_dir='cosmo0',type_dir='pre_r')
    ss = xis.ss
    cov = xis.calc_s_covarianceMatrix(start=dist[0],end=dist[1])  
    xis.calc_sum(n_sub=64,seed=100)


    p_lin =  calc_xi.Decompose_beta('../data/linear_power_z0.15.dat',z=0.15)
    r,xi = p_lin.calc_fourierTransform(r_range=[0.1,250.],r_bin=200)
    f_xi = interpolate.interp1d(r,xi)


    p_lin2 =  calc_xi.fit_to_sigma('../data/linear_power_z0.15.dat',z=0.15)
    r2,xi2 = p_lin2.calc_fourierTransform(r_range=[0.1,250.],r_bin=200)
    f_xi2 = interpolate.interp1d(r2,xi2)

    alpha = np.zeros(64)
    bias = np.zeros(64)
    chi = np.zeros(64)


    alpha2 = np.zeros(64)
    bias2 = np.zeros(64)
    chi2 = np.zeros(64)

    sigma = np.zeros(64)
    sigma2 = np.zeros(64)
    for i,mono in enumerate(xis.monos):
        chis,sigmas,alphas,params,xi_sigma = p_lin.least_squareFit_alpha(ss[dist[0]:dist[1]],mono[dist[0]:dist[1]],cov/4.)
        place = np.where(chis==np.min(chis))
        alpha[i] = alphas[place[1]]
        sigma[i] = sigmas[place[0]]
        param = params[place[0],place[1]]
        chi[i] = np.min(chis)
        bias[i] = np.sqrt(param[0,0])
        print i,100*(alpha[i]-1.),bias[i],chi[i]


        chis2,sigmas2,alphas2,params2,xi_sigma2 = p_lin2.least_squareFit_alpha(xis2.ss[dist[0]:dist[1]],xis2.monos[i,dist[0]:dist[1]],cov2/4.)
        place2 = np.where(chis2==np.min(chis2))
        alpha2[i] = alphas2[place2[1]]
        sigma2[i] = sigmas2[place2[0]]
        param2 = params2[place2[0],place2[1]]
        chi2[i] = np.min(chis2)
        bias2[i] = np.sqrt(param2[0,0])
        print i,100*(alpha2[i]-1.),bias2[i],chi2[i]
        
    print 100*np.mean(alpha-1.),100*np.std(alpha)
   

    print np.shape(np.array((bias,alpha,chi)).T)
    np.savetxt('sigma2,alpha_bias_'+header+'_s_sigmas.txt',np.array((bias,alpha,sigma,chi,bias2,alpha2,sigma2,chi2)).T,header='#bias_pre, alpha_pre, sigma_pre, chi^2_pre, bias_post, alpha_post, sigma_post, chi^2_post',fmt='%f %f %f %f %f %f %f %f')


if __name__=='__main__':
    main()


