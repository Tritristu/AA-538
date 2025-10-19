import numpy as np
import matplotlib.pyplot as plt

# PRE-ASSIGNED PARAMETERS
FACTOR_OF_SAFETY = 2
W = 60000 # N
B = 1.2 # m
NUMBER_OF_STRUTS = 3
FAILURE_STRESS = 150e6 # Pa
YOUNGS_MODULUS = 75e9
DENSITY = 2800 # kg/m^3

# D
objective = lambda D,H: NUMBER_OF_STRUTS*DENSITY*area(D)*length(H)
area = lambda D : np.pi*(D/2)**2
length = lambda H : np.sqrt(H**2 + (1/3)*B**2)
momentInertia = lambda D : np.pi*(D**4)/64
P = lambda H : W*length(H)/(3*H)
stress = lambda  D,H : P(H)/area(D)
P_cr = lambda D,H : (np.pi**2)*YOUNGS_MODULUS*momentInertia(D)/(length(H)**2)
P_allowable = lambda D,H : P_cr(D,H)/FACTOR_OF_SAFETY

D,H = np.meshgrid(np.linspace(0,0.3,300),np.linspace(0,2.5,300))
fig, ax = plt.subplots()
x1 = plt.axvline(x=0.01,color='g',linestyle='-')
x2 = plt.axvline(x=0.2,color='g',linestyle='-.')
y1 = plt.axhline(y=0.01,color='b',linestyle='-')
y2 = plt.axhline(y=2,color='b',linestyle='-.')
CS1 = plt.contour(D,H,P(H)-P_cr(D,H),[0],colors='m')
CS2 = plt.contour(D,H,stress(D,H),[FAILURE_STRESS],colors='r')
CS3 = plt.contour(D,H,objective(D,H),[4.2,10,20,50],colors='k')
shade = plt.imshow( ((H>=0.01) &
                    (H<=2) & 
                    (D>=0.01) & 
                    (D<=0.2)&
                    (P(H)<P_cr(D,H))&
                    (stress(D,H)<FAILURE_STRESS)).astype(int), 
                    extent=(D.min(),D.max(),H.min(),H.max()),origin="lower", cmap="Greys", alpha = 0.3,aspect='auto');

plt.xlim(0,0.3)
plt.ylim(0,2.5)
plt.legend(labels=['Minimum Diameter','Maximum Diameter','Minimum Height','Maximum Height','Allowable load','Buckling Stress','Feasible Area'],loc='center right')
plt.clabel(CS3,[4.2,10,20,50],inline=1,fontsize=8)
plt.xlabel('D')
plt.ylabel('H')
plt.title('Design Space')

moduli = [YOUNGS_MODULUS,YOUNGS_MODULUS*1.1,YOUNGS_MODULUS*1.2,YOUNGS_MODULUS*1.3,YOUNGS_MODULUS*1.4]

plt.figure(2)
plt.plot(moduli,[4.4,4.2,4.1,3.85,3.72],color='b')
plt.xlim([YOUNGS_MODULUS,YOUNGS_MODULUS*1.4])
plt.ylim([0,5])
plt.xlabel("Young's Modulus (Pa)")
plt.ylabel('Mass (kg)')
plt.title('Optimal Design')

plt.figure(3)
plt.plot(moduli,[0.0293,0.02865,0.0282,0.0275,0.027],color='b')
plt.xlim([YOUNGS_MODULUS,YOUNGS_MODULUS*1.4])
plt.ylim([0,0.03])
plt.xlabel("Young's Modulus (Pa)")
plt.ylabel('Diameter (m)')
plt.title('Optimal Design')

plt.figure(4)
plt.plot(moduli,[0.353,0.35,0.36,0.34,0.345],color='b')
plt.xlim(YOUNGS_MODULUS,YOUNGS_MODULUS*1.4)
plt.ylim([0,0.4])
plt.xlabel("Young's Modulus (Pa)")
plt.ylabel('Height (m)')
plt.title('Optimal Design')

plt.show()