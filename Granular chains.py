
# coding: utf-8

# In[1]:

import math
import numpy as np


# In[11]:

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
    
    #energy matrix
    EE = [E]
    #change in q
    dp = np.zeros(qp.shape[0]-1)
    #normal impulse
    P = np.zeros(qp.shape[0]-1)
    PP = [P]
    #contact force
    lamb = np.zeros(qp.shape[0])
    
    #initial values
    for j in xrange(qp.shape[0]-1):
        dp[j] = qp[j] - qp[j+1]
        lamb[j] = (1 + n[j]) ** (n[j]/(n[j] + 1)) * K[j] ** (1/(n[j] + 1)) * E[j] ** (n[j]/(n[j] + 1))
        
    #Integration
    p = 0
    pp = [0]
    #t = 0
    #tt = [0]
    Termination = False
    #termination = true: impact is over
    #termination = false: otherwise
    k = 0
    
    while not Termination:
        #Contacts
        flag = np.zeros(qp.shape[0] - 1)
        #flag[j] = 0: contact does not come into collition
        #flag[j] = 1 contact begins the compression phase
        #flag[j] = 2 contact is already in the impact process
        
        Termination = True
        for j in xrange(qp.shape[0] - 1 ):
            if E[j] == 0:
                if dp[j] <= 0:
                    flag[j] = 0
                else:
                    flag[j] = 1
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
            for j in xrange(qp.shape[0]):
                if d[i] < d[j]:
                    i = j
        else:
            PrimaryContactInVel = False
            for j in xrange(qp.shape[0]):
                if E[i] < E[j]:
                    i = j
            
        #Compute the distributing vector
        Ta = np.zeros(qp.shape[0])
        ni = np.zeros(qp.shape[0] -1)
        Ki = np.zeros(qp.shape[0] -1)
        dpi = np.zeros(qp.shape[0] - 1)
        dP = np.zeros(qp.shape[0] - 1)
        Ai = np.zeros(qp.shape[0] - 1)
        Ei = np.zeros(qp.shape[0] - 1)
        
        for j in xrange(qp.shape[0] - 1):
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
        for j in xrange(qp.shape[0] - 1):
            tmp = dp[j]
            dp[j] = qp[j] - qp[j+1]
            P[j] += dP[j]
            if dp[j] >= 0:
                E[j] += ((tmp + dp[j]) / 2) * dP[j]
            else:
                E[j] += (1 / (es[j] ** 2)) *  ((tmp + dp[j]) / 2) * dP[j]
            lamb[j] += ((1 + n[j]) ** (n[j] / (n[j] + 1))) * (K[j] ** (1 / (n[j] + 1))) * (E[j] ** (n[j] / (n[j] + 1)))
        EE.append(E)
        PP.append(P)
        p += Dp
        pp.append(p)
        #t += Dp/lamb[i]
        #tt.append(t)
        #Advance to next step
        k += 1
    return {'impulse_space':pp, 'momentum':PP, 'Potential':EE}


# In[12]:

qp = np.array([-1.,0.,0.])
M = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
W = np.array([[-1.,0.,0.], [1.,-1.,0.], [0.,1.,-1.]])
K = np.array([1.,1.,1.])
E = np.array([0.5,0.5,0.5])
n = np.array([1.5,1.5,1.5])
es = np.array([1.,1.,1.])
Dp = 1e-6
g = integration(qp, M, W, K, E, n, es, Dp)
print g['impulse_space']


# In[ ]:



