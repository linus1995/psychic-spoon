# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:36:30 2022

@author: linus

Kopplung der Eulergleichungen mit TTM und Zustandsgleichung für angeregtes Gold

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.polynomial.polynomial import Polynomial, polyval



# Parameter & Diskretisierung
small_number = 1e-15
gamma = 1.4

sigma_max = 0.1
Fo_max = 0.5

Nx=90
xmin=0 
xmax=2e-7


dx=(xmax-xmin)/Nx
x = np.arange(xmin-2*dx,xmax+2*dx,dx)

tmax=10e-12 # s


# Anfangsbedingungen
u = np.zeros_like(x)
rho = np.ones_like(x)*19320 # kg/m^3

T_e_0 = 300*np.ones_like(x) # Elektronentemperatur in K
T_ph_0 = 300*np.ones_like(x) # Phononentemperatur in K


# Phasendiagramm / Zustandsgleichungen
I=np.load('Au_PD_inter_Nr500_Nt500.npy',allow_pickle=True).item()  # Phononensystem
       
I_e=np.load('Au_PD_TF_inter.npy',allow_pickle=True).item()  # Elektronensystem

def p_PD(T, rho):
    p = [I["p_ms1"](T[i]/1e3,rho[i]/1e3)[0][0] for i in range(len(T))] # GPa
    return np.array(p)*1e9 # Pa

def p_PD_e(T, rho):
    p = [I_e["p_e"](T[i]/1e3,rho[i]/1e3)[0][0] for i in range(len(T))] # GPa
    return np.array(p)*1e9 # Pa


# Anfangswerte für Druck
p_ph = p_PD(T_ph_0, rho)    
p_e = p_PD_e(T_e_0, rho)                
p = p_ph + p_e



# Thermische Leitfähigkeit

# @njit
def lambda_e_T(T):   
    K = 353     # W/m/K
    b = 0.16
    k_B = 1.380649e-23    # J/K
    u_f = 1.39e6          # Fermigeschwindigkeit in m/s
    m_e = 9.109383e-31    # kg
    e_f = 1/2*m_e*u_f**2  # Fermienergie in J
    T_i = T_e_0   # K
    theta = k_B*T/e_f 
    theta_i = k_B*T_i/e_f
    y = K*(theta**2+0.16)**(5/4)*(theta**2+0.44)*theta/ \
        ((theta**2+0.092)**0.5*(theta**2+b*theta_i))
    return y


# Wärmekapazität aus Temperatur
# @njit
def cp_e_T(T_e):
    lam = 67.6
    y = lam*T_e
    return y


# Wärmekapazität aus Energie
# @njit
def cp_e_epsilon(epsilon_e):
    lam = 67.6
    y = np.sqrt(lam*epsilon_e)
    return y


# Anfangswerte für Energien
cp_ph = 128*np.ones_like(x)*rho        
epsilon_e = T_e_0*cp_e_T(T_e_0)
epsilon_ph = T_ph_0*128*rho



# Laserpuls
kappa = 2
wavelenth = 500e-9
alpha = 4*kappa*np.pi/wavelenth # Absorptionskoeffizient

# @njit
def pulse(t): 
    t_pulse = 10e-15  
    I_0 = 1e16        
    t_0 = 2*t_pulse
    I = I_0*np.exp(-4*np.log(2)*(t-t_0)**2/(t_pulse)**2)      
    return I



# Kopplungsfaktor Elektronen/Phononensystem
data = np.loadtxt("G_Au.dat")
T_dat = data[:,:-1]*1e4
G_dat = data[:,1:]*1e17
T_dat = T_dat[:,0]
G_dat = G_dat[:,0]

coeff = np.polyfit(T_dat,G_dat,27)

def G_values(T_e):
    g = np.where(T_e < np.max(T_dat), polyval(T_e, coeff[::-1]), G_dat[-1])
    return g


# Energieadvektion Zwischenschritt 1
def e1(rho, rho_1, epsilon, F_m, dt_dx, u): 
    epsilon = epsilon /rho
    delta_epsilon = np.zeros_like(x)
    epsilon_adv = np.zeros_like(x)
    F_e = np.zeros_like(x)
    epsilon_1 = np.zeros_like(x)
    
    # Finite Differenzenschema auf versetztem Gitter
    # Flux Limiter nach van Leer
    Au=u[1:-1] > 0
    a = (epsilon[2:] - epsilon[1:-1])*(epsilon[1:-1] - epsilon[:-2])
    delta_epsilon[1:-1] = 2*a / (epsilon[2:] - epsilon[:-2] + small_number) * (a>0)
    
    
    epsilon_adv[1:-1] = ( (epsilon[:-2] + 0.5*(1-dt_dx*u[1:-1])*delta_epsilon[:-2])*Au
                       + (epsilon[1:-1] - 0.5*(1+dt_dx*u[1:-1])*delta_epsilon[1:-1])*(1-Au))
    
    F_e[1:-1] = F_m[1:-1]*epsilon_adv[1:-1]

    epsilon_1[2:-2] = (rho[2:-2]*epsilon[2:-2] - dt_dx*(F_e[3:-1] - F_e[2:-2]))/rho_1[2:-2]   
    
    # Randbedingungen
    epsilon_1[1] = epsilon_1[2]
    epsilon_1[0] = epsilon_1[3]
    epsilon_1[-2] = epsilon_1[-3]
    epsilon_1[-1] = epsilon_1[-4]
    
    return epsilon_1*rho_1


# Energieadvektion Schritt 2
def e2(epsilon_1, rho_1, dt_dx, p_1, u_1):
    
    epsilon_n2 = np.zeros_like(x)
    epsilon_n2[2:-2] = epsilon_1[2:-2] - dt_dx*p_1[2:-2]*(u_1[3:-1] - u_1[2:-2])
    
    #Randbedingungen
    epsilon_n2[1] = epsilon_n2[2]
    epsilon_n2[0] = epsilon_n2[3]
    epsilon_n2[-2] = epsilon_n2[-3]
    epsilon_n2[-1] = epsilon_n2[-4]
    
    return epsilon_n2




# Arrays für neue Zeitschritte
u_n2 = np.zeros_like(x)
rho_1 = np.zeros_like(x)
epsilon_e_new = np.zeros_like(x)
epsilon_ph_new = np.zeros_like(x)


def f(rho, rho_1, u, u_n2, epsilon_e, epsilon_e_new, epsilon_ph, epsilon_ph_new):
    
    # Physikalische Größen
    cp_e = cp_e_epsilon(epsilon_e)      # Wärmekapazität Elektronensystem
    T_e = epsilon_e/cp_e                # Temperatur Elektronensystem
    lam_e = lambda_e_T(T_e)             # Wärmeleitfähigkeit Elektronensystem
    a_e_t=lam_e/cp_e                    
    a_e_t_max = np.max(a_e_t)           
    
    # Stabilitätskriterium
    dt_new_temp_e = Fo_max*dx**2/a_e_t_max    # Zeitschritt aus Fourier-Zahl
    
    if t == 0:
        dt_new = 1e-16
    else:
        dt_new_euler = sigma_max*dx/np.max(u)   # Zeitschritt aus CFL-Zahl
        dt_new = (min([dt_new_temp_e, dt_new_euler]))/2 # neuer Zeitschritt
        
    dt_dx=dt_new/dx


    delta_rho = np.zeros_like(x)
    rho_adv = np.zeros_like(x)
    F_m = np.zeros_like(x)
    delta_rho = np.zeros_like(x)
    rho_adv = np.zeros_like(x)

    # Flux Limiter nach van Leer
    
    a = (rho[2:] - rho[1:-1])*(rho[1:-1] - rho[:-2])
    delta_rho[1:-1] = 2*a / (rho[2:] - rho[:-2] + small_number) * (a>0)
    
    Au=u[1:-1] > 0
    rho_adv[1:-1] = ( (rho[:-2] + 0.5*(1-dt_dx*u[1:-1])*delta_rho[:-2])*Au
                   + (rho[1:-1] - 0.5*(1+dt_dx*u[1:-1])*delta_rho[1:-1])*(1-Au))

    F_m[1:-1]=rho_adv[1:-1]*u[1:-1]
    
    # Lineare Transportgleichung Dichte
    rho_1[2:-2] = rho[2:-2] - dt_dx*(F_m[3:-1] - F_m[2:-2])

    #Randbedingungen 
    rho_1[0] = rho_1[3]
    rho_1[1] = rho_1[2]
    rho_1[-2] = rho_1[-3]
    rho_1[-1] = rho_1[-4]
    
    
    
    # Impulsgleichung
    delta_u = np.zeros_like(x)
    u_average = np.zeros_like(x)
    u_adv = np.zeros_like(x)
    u_1 = np.zeros_like(x)

    F_I = np.zeros_like(x)
    rho_average_1 = np.zeros_like(x)
    rho_average = np.zeros_like(x)
    
    a = (u[2:] - u[1:-1])*(u[1:-1] - u[:-2])
    delta_u[1:-1] = 2*a / (u[2:] - u[:-2] + small_number) * (a>0)
    
    u_average[1:-1] = 0.5*(u[1:-1] + u[2:])
    u_av_i=u_average[1:-1]

    Au_av=u_av_i > 0
    u_adv[1:-1] = ( (u[1:-1] + 0.5*(1-dt_dx*u_av_i)*delta_u[1:-1])*Au_av
                    + (u[2:] - 0.5*(1+dt_dx*u_av_i)*delta_u[2:])*(1-Au_av))

    F_I[1:-1]=0.5*(F_m[1:-1] + F_m[2:])*u_adv[1:-1]

    rho_average[1:-1] = 0.5*(rho[:-2] + rho[1:-1]) 
    rho_average_1[1:-1] = 0.5*(rho_1[:-2] + rho_1[1:-1]) 
    
    
    # Zwischenschritt Impulsgleichung
    u_1[3:-2] = (u[3:-2]*rho_average[3:-2] - dt_dx*(F_I[3:-2]-F_I[2:-3]))/ \
        (rho_average_1[3:-2] + small_number)       
    
    u_1[2] = 0          
    u_1[1] = -u_1[3]
    u_1[-2] = 0
    u_1[-1] = -u_1[-3]

    

    
    # Zwischenschritt Energie
    epsilon_e_1 = e1(rho, rho_1, epsilon_e, F_m, dt_dx, u) 
    epsilon_ph_1 = e1(rho, rho_1, epsilon_ph, F_m, dt_dx, u)
    cp_e_1 = cp_e_epsilon(epsilon_e_1)      # Wärmekapazität Elektronen
    
    
    
    # Zwischenschritt Temperatur
    T_e = epsilon_e_1/cp_e_1   
    T_ph = epsilon_ph_1/cp_ph 
    lam_e = lambda_e_T(T_e)
    
    
    # Zwischenschritt Druck
    #p_1 = (gamma - 1)*rho_1*epsilon_ph_1        # ideales Gas
    p_1_ph = p_PD(T_ph, rho_1)                   # Druck aus Phasendiagramm (Elektronensystem)
    p_1_e = p_PD_e(T_e, rho_1)                   # Druck aus Phasendiagramm (Phononensystem)
    p_1 = p_1_ph + p_1_e
    
    
    

    # neue Geschwindigkeit
    u_n2[3:-2] = u_1[3:-2] - dt_dx*(p_1[3:-2] - p_1[2:-3]) / \
        (rho_average_1[3:-2] + small_number)

    u_n2[2] = 0         
    u_n2[1] = -u_n2[3]
    u_n2[-2] = 0
    u_n2[-1] = -u_n2[-3]
    
     
    # Quelle
    source = pulse(t)*np.exp(-alpha*x)*alpha
    G = G_values(T_e) 
    

    # neue Energie
    epsilon_e_new[1:-1] = e2(epsilon_e_1, rho_1, dt_dx, p_1, u_1)[1:-1] \
        + dt_new*(((lam_e[2:]+lam_e[1:-1])*(T_e[2:] - T_e[1:-1]) \
                   - (lam_e[1:-1]+lam_e[:-2])*(T_e[1:-1] - T_e[:-2]))/(2*dx**2))  \
            +     dt_new*(source[1:-1] - G[1:-1]*(T_e[1:-1] - T_ph[1:-1]))
   

    # neue Energie
    epsilon_ph_new[1:-1] = (e2(epsilon_ph_1, rho_1, dt_dx, p_1, u_1)[1:-1] \
                            + dt_new*((G[1:-1]*(T_e[1:-1] - T_ph[1:-1]))))
    
    # Randbedingungen
    epsilon_ph_new[1] = epsilon_ph_new[2]
    epsilon_ph_new[0] = epsilon_ph_new[3]
    epsilon_ph_new[-2] = epsilon_ph_new[-3]
    epsilon_ph_new[-1] = epsilon_ph_new[-4]
    
    epsilon_e_new[1] = epsilon_e_new[2]
    epsilon_e_new[0] = epsilon_e_new[3]
    epsilon_e_new[-2] = epsilon_e_new[-3]
    epsilon_e_new[-1] = epsilon_e_new[-4]
    
    
    # neue Temperatur
    T_e = epsilon_e_new/cp_e_epsilon(epsilon_e_new)
    T_ph = epsilon_ph_new/cp_ph 
    
    return p_1, p_1_e, p_1_ph, dt_new, T_e, T_ph




time = []
t = 0
count = 0

while t <= tmax:


    [p_1, p_1_e, p_1_ph, dt_new, T_e, T_ph] \
    = f(rho, rho_1, u, u_n2, epsilon_e, epsilon_e_new, epsilon_ph, epsilon_ph_new)
    
    #Zirkelverweise
    rho_help=rho
    rho=rho_1
    rho_1=rho_help
    
    u_help=u
    u=u_n2
    u_n2=u_help
    
    epsilon_e_help=epsilon_e
    epsilon_e=epsilon_e_new
    epsilon_e_new=epsilon_e_help
    
    epsilon_ph_help=epsilon_ph
    epsilon_ph=epsilon_ph_new
    epsilon_ph_new=epsilon_ph_help

    t += dt_new

    print(t)



# Graphische Darstellung für Präsentation

fig, ax = plt.subplots(1,2)

ax[0].plot(x*1e9, rho-19320)
ax[1].plot(x*1e9,T_e, label = "Elektronen")
ax[1].plot(x*1e9,T_ph, label = "Phononen")

ax[0].set_ylabel(r"$\bigtriangleup \rho$ in $\dfrac{kg}{m^{3}}$", fontsize = 30)
ax[0].set_xlabel("x in nm",  fontsize = 30)
ax[1].set_ylabel("T in K", fontsize = 30)
ax[1].set_xlabel("x in nm",  fontsize = 30)
    
ax[0].tick_params(axis="x", labelsize=24)
ax[0].tick_params(axis="y", labelsize=24)

ax[1].tick_params(axis="x", labelsize=24)
ax[1].tick_params(axis="y", labelsize=24)
ax[0].set_title("Dichteänderung bei " + str(tmax*1e12) + "ps", fontsize=34)
ax[0].set_ylim([np.min(rho)-19320, np.max(rho)-19320])
ax[1].set_title("Temperatur bei " + str(tmax*1e12) + "ps", fontsize=34)
ax[1].legend(fontsize=20)
plt.subplots_adjust(left=0.09,
                    bottom=0.09, 
                    right=0.99, 
                    top=0.95, wspace = 0.25, hspace = 0.33)


fig, ax = plt.subplots(1,1)

ax.plot(x*1e9, p_1*1e-9, label = "Gesamt")
ax.plot(x*1e9, p_1_e*1e-9, label = "Elektronen")
ax.plot(x*1e9, p_1_ph*1e-9, label = "Phononen")

ax.set_xlabel("x in nm",  fontsize = 26)
ax.set_ylabel("p in GPa",  fontsize = 26)
ax.legend(fontsize=20)
ax.set_title("Druck bei " + str(tmax*1e12) + "ps", fontsize=36)
ax.tick_params(axis="x", labelsize=22)
ax.tick_params(axis="y", labelsize=22)
plt.subplots_adjust(left=0.08,
                    bottom=0.09, 
                    right=0.99, 
                    top=0.95, wspace = 0.25, 
                    hspace = 0.33)