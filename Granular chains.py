
# coding: utf-8

# In[1]:

import math
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

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


# In[30]:

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
    #change in q
    dp = np.zeros(qp.shape[0]-1)
    ff = [dp]
    #normal impulse
    P = np.zeros(qp.shape[0]-1)
    PP = [P]
    #contact force
    lamb = np.zeros(qp.shape[0]-1)
    lambb = [lamb]
    
    #initial values
    for j in xrange(qp.shape[0]-1):
        dp[j] = qp[j] - qp[j+1]
        lamb[j] = (1 + n[j]) ** (n[j]/(n[j] + 1)) * K[j] ** (1/(n[j] + 1)) * E[j] ** (n[j]/(n[j] + 1))
        
    #Integration
    
    #impulse
    p = 0
    #vector that stores the impulse space
    pp = [p]
    #time
    t = 0
    #vector that stores the time space
    tt = [t]
    Termination = False
    #termination = true: impact is over
    #termination = false: otherwise
    k = 0
    
    #Compute the distributing vector Ta
    Ta = np.zeros(qp.shape[0])
    ni = np.zeros(qp.shape[0] -1)
    Ki = np.zeros(qp.shape[0] -1)
    dpi = np.zeros(qp.shape[0] - 1)
    dP = np.zeros(qp.shape[0] - 1)
    Ai = np.zeros(qp.shape[0] - 1)
    Ei = np.zeros(qp.shape[0] - 1)
    
    while (not Termination) | k < 40000:
        #Contacts
        flag = np.zeros(qp.shape[0] - 1)
        #flag[j] = 0: contact does not come into collition
        #flag[j] = 1 contact begins the compression phase
        #flag[j] = 2 contact is already in the impact process
        
        Termination = True
        for j in xrange(qp.shape[0] - 1 ):
            if E[j] <= 1e-36:
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
            for j in xrange(qp.shape[0]-1):
                if d[i] < d[j]:
                    i = j
        else:
            PrimaryContactInVel = False
            for j in xrange(qp.shape[0]-1):
                if E[i] < E[j]:
                    i = j
        
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
        
        #Appends variables in memory
        EE = np.concatenate((EE, [E]), axis=0)
        PP =np.concatenate((PP, [P]), axis=0)
        p += Dp
        pp = np.concatenate((pp, [p]), axis=0)
        qq = np.concatenate((qq, [qp]), axis=0)
        ff = np.concatenate((ff, [dp]), axis=0)
        #if lamb[i] == 0:
        #    print lamb.shape
        #    t += (dP[i] ** (1./(1+n[i]))) / ((1 + n[i]) ** (n[i] / (n[i] + 1)) * K[i] ** (1 / (n[i] + 1)) * dp[i] ** (n[i] / (n[i] + 1)))
        #else:
        t += Dp/lamb[i]
        tt.append(t)
        lambb = np.concatenate((lambb, [lamb]), axis=0)
        #Advance to next step
        k += 1
        #print qq.shape
        if k%1000 == 0:
            print k
    return {'time':tt, 'force':lambb, 'relative':ff, 'impulse_space':pp, 'momentum':PP, 'Potential':EE, 'velocity':qq}


# In[31]:

#Book example
qp = np.array([-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
M = 2.094e-6*7780.*np.eye(10)
W = np.array([[-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.], [1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.], [0.,1.,-1.,0.,0.,0.,0.,0.,0.,0.], [0.,0.,1.,-1.,0.,0.,0.,0.,0.,0.], [0.,0.,0.,1.,-1.,0.,0.,0.,0.,0.], [0.,0.,0.,0.,1.,-1.,0.,0.,0.,0.], [0.,0.,0.,0.,0.,1.,-1.,0.,0.,0.], [0.,0.,0.,0.,0.,0.,1.,-1.,0.,0.], [0.,0.,0.,0.,0.,0.,0.,1.,-1.,0.], [0.,0.,0.,0.,0.,0.,0.,0.,1.,-1.]])
K = np.array([21031894.0045,21031894.0045,21031894.0045,21031894.0045,21031894.0045,21031894.0045,21031894.0045,21031894.0045,21031894.0045])
E = np.array([1e-10,1e-10,1e-10,1e-10,1e-10,1e-10,1e-10,1e-10,1e-10])
n = np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5])
es = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.])
Dp = 1.
g = integration(qp, M, W, K, E, n, es, Dp)
#print g['velocity']


# In[22]:

#Generic example
qp0 = np.array([1.,0.,0.,0.,0.])
M0 = np.array([[1.,0.,0.,0.,0.], [0.,1.,0.,0.,0.], [0.,0.,1.,0.,0.], [0.,0.,0.,1.,0.], [0.,0.,0.,0.,1e20]])
W0 = np.array([[-1.,0.,0.,0.,0.], [1.,-1.,0.,0.,0.], [0.,1.,-1.,0.,0.], [0.,0.,1.,-1.,0.], [0.,0.,0.,1.,-1.]])
K0 = np.array([1.,1.,1.,1.])
E0 = np.array([1e-10,1e-10,1e-10,1e-10])
n0 = np.array([1.5,1.5,1.5,1.5])
es0 = np.array([1.,1.,1.,1.])
Dp0 = 1e-3
g0 = integration(qp0, M0, W0, K0, E0, n0, es0, Dp0)
#print g['velocity']


# In[ ]:

plt.plot(g['impulse_space'], g['velocity'])
plt.show()
print g0['velocity'].shape


# In[ ]:



