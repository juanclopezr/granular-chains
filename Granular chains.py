
# coding: utf-8

# In[3]:

import math
import numpy as np
import matplotlib.pyplot as plt


# In[4]:

def step_size(M, qp, n):
    """
    param M:  Square diagonal Mass matrix of dimensions dxd.
    param qp: The relative speed vector of length d.
    param n:  The number of steps for the integration process.
    
    returns   The step size, DP, t the primary contact point for the algorithm in a floating number.
    """
    Pest = 0.
    for i in xrange(M.shape[0]):
        for j in xrange(M.shape[0]):
            Pest += M[i][j] * qp[j]
    DP = Pest / n
    return DP


# In[25]:

#LZB Multiple impact models
def integration(qp, M, W, E, K, n, es, Dp):
    """
    param qp:the change in time of the generalized coordinate vector.
    param M: the mass matrix.
    param W: the matrix of the gradient of the constraints
    param E: the vector of the initial potential energy at all contacts.
    param K: the stiffness vector.
    param n: the vector of contact elasticity coefficients.
    param es: vector of energetic restitution coeffitients.
    param DP: the size of the impulse step for the integration process.
    """
    #variable initialization
    qq = [qp]
    #energy matrix
    EE = [E]
    #reltive speed
    dp = np.zeros(qp.shape[0]-1)
    #relative speed matrux
    ff = [dp]
    #normal impulse
    P = np.zeros(qp.shape[0]-1)
    #normal impulse matrix
    PP = [P]
    #contact force
    #lamb = np.zeros(qp.shape[0]-1)
    #contact force matrix
    #lambb = [lamb]
    
    #initial values
    #for j in xrange(qp.shape[0]-1):
    #    dp[j] = qp[j] - qp[j+1]
    #    lamb[j] = (1 + n[j]) ** (n[j]/(n[j] + 1)) * K[j] ** (1/(n[j] + 1)) * E[j] ** (n[j]/(n[j] + 1))
        
    #Integration
    
    #impulse
    p = 0
    #vector that stores the impulse space
    pp = [p]
    #time
    #t = 0
    #vector that stores the time space
    #tt = [t]
    Termination = False
    #termination = true: impact is over
    #termination = false: otherwise
    k = 0
    
    #Compute the distributing vector Ta
    Ta = np.zeros(qp.shape[0])
    ni = np.zeros(qp.shape[0] - 1)
    Ki = np.zeros(qp.shape[0] - 1)
    dpi = np.zeros(qp.shape[0] - 1)
    dP = np.zeros(qp.shape[0] - 1)
    Ai = np.zeros(qp.shape[0] - 1)
    Ei = np.zeros(qp.shape[0] - 1)
    
    #Contacts
    flag = np.zeros(qp.shape[0] - 1)
    #flag[j] = 0: contact does not come into collition
    #flag[j] = 1 contact begins the compression phase
    #flag[j] = 2 contact is already in the impact process
    
    #for speed purposes
    
    rang = range(qp.shape[0] - 1)
    while (not Termination) | k < 40000:
        
        Termination = True
        for j in rang:
            if E[j] == 0:
                if dp[j] <= 0:
                    flag[j] = 0
                else:
                    flag[j] = 1
                    Termination = False
            else:
                flag[j] = 2
                Termination = False
        
        #Condition of contact
        PrimaryContactInVel = True
        #True: the primary contact is selected according to the relative velocity
        #False: the primary contct is selected according to the potential energy
        
        i = 0
        maxE = max(E)
        if maxE == 0:
            PrimaryContactInVel = True
            for j in rang:
                if dp[i] < dp[j]:
                    i = j
        else:
            PrimaryContactInVel = False
            for j in rang:
                if E[i] < E[j]:
                    i = j
        
        for j in rang:
            ni[j] = ((1 + n[j]) ** (n[j] / (n[j] + 1))) / ((1 + n[i]) ** (n[i] / (n[i] + 1)))
            Ki[j] = (K[j] ** (1 / (1 + n[j]))) / (K[i] ** (1 / (1 + n[i])))
            
            if PrimaryContactInVel:
                dpi[j] = (d[j] ** (n[j] / (1 + n[j]))) / (d[i] ** (n[i] / (1 + n[i])))
                Ta[j] = ((ni[j] * Ki[j] * dpi[j]) ** (n[j] + 1)) * (Dp ** ((n[j] - n[i]) / n[i] + 1))
            else:
                if flag[j] == 0:
                    dP[j] = 0
                elif flag[j] == 1:
                    Ai[j] = (dp[j] * Dp) ** (n[j] / (n[j] + 1)) / (E[i] ** (n[i] / (n[i] + 1)))
                    Ta[j] = (ni[j] * Ki[j] * Ai[j]) ** (n[j] + 1)
                    dP[j] = Dp*Ta[j]
                elif flag[j] ==2:
                    Ei[j] = (E[j] ** (n[j] / (n[j] + 1))) / (E[i] ** (n[i] / (n[i] + 1)))
                    Ta[j] = ni[j] * Ki[j] * Ei[j]
                    dP[j] = Dp*Ta[j]
        
        qp += Dp * np.dot(np.dot(np.linalg.inv(M), W), Ta)
        for j in rang:
            tmp = dp[j]
            dp[j] = qp[j] - qp[j+1]
            P[j] += dP[j]
            if dp[j] >= 0:
                E[j] += ((tmp + dp[j]) / 2) * dP[j]
            else:
                E[j] += (1 / (es[j] ** 2)) *  ((tmp + dp[j]) / 2) * dP[j]
            #lamb[j] += ((1 + n[j]) ** (n[j] / (n[j] + 1))) * (K[j] ** (1 / (n[j] + 1))) * (E[j] ** (n[j] / (n[j] + 1)))
        
        #Appends variables in memory
        EE = np.concatenate((EE, [E]), axis=0)
        PP =np.concatenate((PP, [P]), axis=0)
        p += Dp
        pp = np.concatenate((pp, [p]), axis=0)
        qq = np.concatenate((qq, [qp]), axis=0)
        ff = np.concatenate((ff, [dp]), axis=0)
        #lambb = np.concatenate((lambb, [lamb]), axis=0)
        #Advance to next step
        k += 1
        #print qq.shape
        if k%1000 == 0:
            print k / 1000
        for j in rang:
            if E[j] < 0:
                print E
    return {'relative':ff, 'impulse_space':pp, 'momentum':PP, 'Potential':EE, 'velocity':qq}


# In[26]:

#Book example
qp1 = np.array([10.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
M1 = np.multiply(0.0183311931337,np.eye(10))
W1 = np.array([[-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.], [1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.], [0.,1.,-1.,0.,0.,0.,0.,0.,0.,0.], [0.,0.,1.,-1.,0.,0.,0.,0.,0.,0.], [0.,0.,0.,1.,-1.,0.,0.,0.,0.,0.], [0.,0.,0.,0.,1.,-1.,0.,0.,0.,0.], [0.,0.,0.,0.,0.,1.,-1.,0.,0.,0.], [0.,0.,0.,0.,0.,0.,1.,-1.,0.,0.], [0.,0.,0.,0.,0.,0.,0.,1.,-1.,0.], [0.,0.,0.,0.,0.,0.,0.,0.,1.,-1.]])
K1 = np.multiply(10515947.0023, np.ones(9))
E1 = np.multiply(1e-10, np.ones(9))
n1 = np.multiply(1.5, np.ones(9))
es1 = np.ones(9)
Dp1 = 10.
#g = integration(qp, M, W, K, E, n, es, Dp)
#print g['velocity']


# In[16]:

#Generic example
qp0 = np.array([1.,0.,0.,0.,0.,0.,0.])
M0 = np.array([[1.,0.,0.,0.,0.,0.,0.], [0.,1.,0.,0.,0.,0.,0.], [0.,0.,1.,0.,0.,0.,0.], [0.,0.,0.,1.,0.,0.,0.], [0.,0.,0.,0.,1.,0.,0.], [0.,0.,0.,0.,0.,1.,0.], [0.,0.,0.,0.,0.,0.,1e20]])
W0 = np.array([[-1.,0.,0.,0.,0.,0.,0.], [1.,-1.,0.,0.,0.,0.,0.], [0.,1.,-1.,0.,0.,0.,0.], [0.,0.,1.,-1.,0.,0.,0.], [0.,0.,0.,1.,-1.,0.,0.], [0.,0.,0.,0.,1.,-1.,0.], [0.,0.,0.,0.,0.,1.,-1.]])
K0 = np.array([1.,1.,1.,1.,1.,1.])
E0 = np.array([1e-10,1e-10,1e-10,1e-10,1e-10,1e-10])
n0 = np.array([1.5,1.5,1.5,1.5,1.5,1.5])
es0 = np.array([1.,1.,1.,1.,1.,1.])
Dp0 = 1e-3
g0 = integration(qp0, M0, W0, K0, E0, n0, es0, Dp0)
#print g['velocity']


# In[27]:

g = integration(qp1, M1, W1, K1, E1, n1, es1, Dp1)
plt.plot(g['impulse_space'], g['Potential'])
plt.show()
print g['velocity'].shape


# In[68]:

c = 203e6 / (0.91 * 2.)
v = (0.01 ** 3.) * (math.pi * 3. * 7780.) / 4.
o = ((0.005 ** 0.5) * c * 4.) / (3. * 0.91)
print v
print o
#print K
#print 11555985.7168 * np.ones(9)
print K1


# In[ ]:



