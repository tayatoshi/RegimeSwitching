import numpy as np
import pandas as pd
from scipy import optimize
from datetime import datetime
import itertools

class Regimeswitching(object):
    """
    State-Space model
    (a_t)' = c_{s_t}} + F{S_t}((a_t-1)' - c_{s_{t-1}}) + u_t
    (y_t)' = H{S_t}(a_t)' +A{S_t}(x_t)' + e_t
    (a_t)' J * 1 matrix
    (y_t)' D * 1 matrix

    TRANC RATE(if M = 2):
             | S_t = 0 | S_t = 1
    ------------------------------
    S_t-1 = 0|    q    |   1-q
    ------------------------------
    S_t-1 = 1|   1-p   |    p
    ------------------------------
    """
    def __init__(self,y,a0,p0,x=None):
        self.len = len(y)
        self.y = y# T * D
        if x == None:
            self.x = np.zeros(self.len).reshape(self.len,1)
        else:
            self.x = x# T * K
        self.a0 = a0# a_{0|0}^{j}/M * J
        self.p0 = p0# p_{0|0}^{j}/M * J * J
        self.M = self.a0.shape[0]
        self.J = self.a0.shape[1]
        self.D = self.y.shape[1]
        self.K = self.x.shape[1]
        # ===========================
        # PARAMATERS
        # ===========================
        self.TRANCE = 0.5 * np.ones([2,2])# Pr(S_t|S_t-1)/M * M
        self.pi0 = np.array([(1 - self.TRANCE[1,1])/(2 - self.TRANCE[0,0] - self.TRANCE[1,1]),
                             (1 - self.TRANCE[0,0])/(2 - self.TRANCE[0,0] - self.TRANCE[1,1])])# [pr(s_0=0|omega_0)=1-p/2-p-q, pr(s_0=1|omega_0)=1-q/2-p-q]
        # self.c = np.ones([self.M,self.J])# constant: M * J
        self.c = np.array([30,60])# constant: M * J
        self.F = np.ones([self.M,self.J,self.J])*0.7# M * J * J
        self.H = np.ones([self.M,self.D,self.J])# M * D * J
        self.A = np.zeros([self.M,self.D,self.K])# M * D * K
        self.Q = np.ones([self.M,self.J,self.J])#state error variance matrix: J * J
        self.R = np.ones([self.M,self.D,self.D])#observation error variance matrix: D * D

        self.a_predict = np.zeros([self.len,self.M,self.M,self.J])# a_{t|t-1}^{i,j}: T * M * M * J
        self.a_filter = np.zeros([self.len,self.M,self.M,self.J])# a_{t|t}^{i,j}: T * M * M * J
        self.p_predict = np.zeros([self.len,self.M,self.M,self.J,self.J])#p_{t|t-1}^{i,j} T * M * M * J * J
        self.p_filter = np.zeros([self.len,self.M,self.M,self.J,self.J])# p_{t|t-1}^{i,j}T * M * J * J
        self.eta = np.zeros([self.len,self.M,self.M,self.D])# T * M * M * D
        self.f = np.zeros([self.len,self.M,self.M,self.D,self.D])# T * M * M * D * D
        self.joint_probability = np.zeros([self.len,self.M,self.M]) #Pr(S_t,S_t-1|omega_t-1)
        self.joint_filter_probability = np.zeros([self.len,self.M,self.M])
        self.marginal_probability = np.zeros([self.len,self.M]) # T*M :Pr(S_t|omega_t)
        self.marginal_likelihood = np.zeros(self.len)# f(y_t|omega_t-1)
        self.marginal_loglikelihood = np.zeros(self.len)# f(y_t|omega_t-1)
        # self.log_likilihood = np.zeros(self.len)
        self.a_marginal = np.zeros([self.len,self.M,self.J])# a_{t|t}^{j}:T * M * J
        self.p_marginal = np.zeros([self.len,self.M,self.J,self.J])# T * M * J * J
        return print("OK")

    def fit(self):
        start = datetime.now()
        Trance = self.TRANCE
        result = self.par_opt()
        print("time:{}".format(datetime.now() - start))
        return {'LogLikelihood':np.sum(self.marginal_loglikelihood),'trance':self.TRANCE, 'marginal_a':self.a_marginal, 'marginal_probability':self.marginal_probability,'c':self.c}

    def predict(self,t):
        if t==0:
            # for i in range(self.M):
            #     for j in range(self.M):
            #         self.a_predict[t,i,j] = self.c[j].reshape(self.J,1) + self.F[j] @ (self.a0[i].reshape(self.J,1) - self.c[i])#a_{1|0}:J * 1
            #         self.p_predict[t,i,j] = self.F[j] @ self.p0[i].reshape(self.J,1) @ self.F[j].T + self.Q[j]#p_{1|0}
            for i,j in itertools.product(range(self.M),range(self.M)):
                self.a_predict[t,i,j] = self.c[j].reshape(self.J,1) + self.F[j] @ (self.a0[i].reshape(self.J,1) - self.c[i])#a_{1|0}:J * 1
                self.p_predict[t,i,j] = self.F[j] @ self.p0[i].reshape(self.J,1) @ self.F[j].T + self.Q[j]#p_{1|0}
        else:
            # for i in range(self.M):
            #     for j in range(self.M):
            #         self.a_predict[t,i,j] = self.c[j] + self.F[j] @ (self.a_marginal[t-1,i] - self.c[i])#a_{t|t-1}:J * 1
            #         self.p_predict[t,i,j] = self.F[j] @ self.p_marginal[t-1,i] @ self.F[j].T + self.Q[j]#p_{t|t-1}
            for i,j in itertools.product(range(self.M),range(self.M)):
                self.a_predict[t,i,j] = self.c[j].reshape(self.J,1) + self.F[j] @ (self.a0[i].reshape(self.J,1) - self.c[i])#a_{1|0}:J * 1
                self.p_predict[t,i,j] = self.F[j] @ self.p0[i].reshape(self.J,1) @ self.F[j].T + self.Q[j]#p_{1|0}

    def residuals(self,t):
        # for i in range(self.M):
        #     for j in range(self.M):
        #         self.eta[t,i,j] = self.y[t].T - self.H[j] @ self.a_predict[t,i,j] - self.A[j] @ self.x[t].T# D * 1
        #         self.f[t,i,j] = self.H[j] @ self.p_predict[t,i,j] @ self.H[j].T + self.R[j]# D * D
        for i,j in itertools.product(range(self.M),range(self.M)):
            self.eta[t,i,j] = self.y[t].T - self.H[j] @ self.a_predict[t,i,j] - self.A[j] @ self.x[t].T# D * 1
            self.f[t,i,j] = self.H[j] @ self.p_predict[t,i,j] @ self.H[j].T + self.R[j]# D * D

    def filtering(self,t):
        # for i in range(self.M):
        #     for j in range(self.M):
        #         self.a_filter[t,i,j] = self.a_predict[t,i,j] + self.p_predict[t,i,j] @ self.H[j].T @ np.linalg.inv(self.f[t,i,j]) @ self.eta[t,i,j]
        #         self.p_filter[t,i,j] = self.p_predict[t,i,j] @ (np.eye(self.J) - self.H[j].T @ np.linalg.inv(self.f[t,i,j]) @ self.H[j] @ self.p_predict[t,i,j].T)
        for i,j in itertools.product(range(self.M),range(self.M)):
            self.a_filter[t,i,j] = self.a_predict[t,i,j] + self.p_predict[t,i,j] @ self.H[j].T @ np.linalg.inv(self.f[t,i,j]) @ self.eta[t,i,j]
            self.p_filter[t,i,j] = self.p_predict[t,i,j] @ (np.eye(self.J) - self.H[j].T @ np.linalg.inv(self.f[t,i,j]) @ self.H[j] @ self.p_predict[t,i,j].T)

    def HamiltonFilter(self,t):
        # Pr(S_t,S_t-1|omega_t-1)
        if t == 0:
            # for i in range(self.M):
            #     for j in range(self.M):
            #         self.joint_probability[t,i,j] = self.TRANCE[i,j] * self.pi0[i]
            for i,j in itertools.product(range(self.M),range(self.M)):
                self.joint_probability[t,i,j] = self.TRANCE[i,j] * self.pi0[i]
        elif t  > 0:
            # for i in range(self.M):
            #     for j in range(self.M):
            #         self.joint_probability[t,i,j] = self.TRANCE[i,j] * self.marginal_probability[t-1,i]
            for i,j in itertools.product(range(self.M),range(self.M)):
                self.joint_probability[t,i,j] = self.TRANCE[i,j] * self.marginal_probability[t-1,i]
        # f(y_t|omega_t-1)
        # for i in range(self.M):
        #     for j in range(self.M):
        #         self.marginal_likelihood[t] = self.marginal_likelihood[t] + self.get_likeihood(t,i,j) * self.joint_probability[t,i,j]
        for i,j in itertools.product(range(self.M),range(self.M)):
            self.marginal_likelihood[t] = self.marginal_likelihood[t] + self.get_likeihood(t,i,j) * self.joint_probability[t,i,j]
            self.marginal_loglikelihood[t] = self.marginal_loglikelihood[t] + self.get_likeihood(t,i,j,log=True) * self.joint_probability[t,i,j]
        # Pr(S_t,S_t-1|omega_t)
        # for i in range(self.M):
        #     for j in range(self.M):
        #         self.joint_filter_probability[t,i,j] = (self.get_likeihood(t,i,j) * self.joint_probability[t,i,j])/self.marginal_likelihood[t]
        for i,j in itertools.product(range(self.M),range(self.M)):
            self.joint_filter_probability[t,i,j] = (self.get_likeihood(t,i,j) * self.joint_probability[t,i,j])/self.marginal_likelihood[t]
        #Pr(S_t|omega_t)
        self.marginal_probability[t] = np.sum(self.joint_filter_probability[t],axis=0)

    def marginal_estimate(self,t):
        for j in range(self.M):
            if t==0:
                denominator = self.pi0[j]
            else:
                if self.marginal_probability[t,j]==0:
                    denominator = 0.0001
                else:
                    denominator = self.marginal_probability[t,j]
                # denominator = self.marginal_probability[t,j]
            a_molecule = np.zeros((self.a_filter[t]).shape)
            # for ii in range(self.M):
            #     for jj in range(self.M):
            #         a_molecule[ii,jj] = self.a_filter[t,ii,jj] * self.joint_filter_probability[t,ii,jj]#分子
            for ii,jj in itertools.product(range(self.M),range(self.M)):
                a_molecule[ii,jj] = self.a_filter[t,ii,jj] * self.joint_filter_probability[t,ii,jj]#分子
            self.a_marginal[t,j] = np.sum(a_molecule,axis=0)[j]/denominator# a_{t|t}^{j}

            p_molecule = np.zeros((self.p_filter[t]).shape)
            # for ii in range(self.M):
            #     for jj in range(self.M):
            #         p_molecule[ii,jj] = self.joint_filter_probability[t,ii,jj] * \
            #                             (self.p_filter[t,ii,jj] + (self.a_filter[t,ii,jj] - self.a_marginal[t,jj]) @ (self.a_filter[t,ii,jj] - self.a_marginal[t,jj]).T)
            for ii,jj in itertools.product(range(self.M),range(self.M)):
                p_molecule[ii,jj] = self.joint_filter_probability[t,ii,jj] * \
                                    (self.p_filter[t,ii,jj] + (self.a_filter[t,ii,jj] - self.a_marginal[t,jj]) @ (self.a_filter[t,ii,jj] - self.a_marginal[t,jj]).T)
            self.p_marginal[t,j] = np.sum(p_molecule,axis = 0)[j]/denominator# p_{t|t}^{j}

    def get_likeihood(self,t,i,j,log = False):
        # normal distribution
        # f(y_t|S_t,S_t-1,omegta_t-1)
        if log == False:
            # mu = self.H[j] @ self.a_predict[t,i,j] + self.A[j] @ self.x[t].T #D * 1
            left = 1/(((2*np.pi)**(self.D/2))*np.sqrt(np.linalg.det(self.f[t,i,j])))
#=================================================ここがおかしい↓
            right = np.exp((-1/2)*(self.eta[t,i,j].T @ np.linalg.inv(self.f[t,i,j]) @ self.eta[t,i,j]))#expないが大きすぎてが0になる
            Likelihood = left * right
        else:
            log_left = -(self.D/2)*np.log(2*np.pi) - (1/2)*np.log(np.linalg.det(self.f[t,i,j]))
            log_right = (-1/2)*(self.eta[t,i,j].T @ np.linalg.inv(self.f[t,i,j]) @ self.eta[t,i,j])
            Likelihood = log_left + log_right
        return Likelihood

    def model(self,tr):
        p = np.round(1/(1+np.exp(-tr[0])),5)
        q = np.round(1/(1+np.exp(-tr[1])),5)
        self.TRANCE = np.array([[p,1-p],
                                [1-q,q]])
        self.pi0 = np.array([(1 - self.TRANCE[1,1])/(2 - self.TRANCE[0,0] - self.TRANCE[1,1]),
                             (1 - self.TRANCE[0,0])/(2 - self.TRANCE[0,0] - self.TRANCE[1,1])])# [pr(s_0=0|omega_0)=1-p/2-p-q, pr(s_0=1|omega_0)=1-q/2-p-q]
        self.c = np.array([tr[2],tr[3]])
        beta = np.tanh(tr[4])
        self.F = np.ones([self.M,self.J,self.J])*beta# M * J * J
        self.a0 = np.array([tr[5],tr[6]])
        for t in range(self.len):
            self.predict(t)
            self.residuals(t)
            self.filtering(t)
            self.HamiltonFilter(t)
            self.marginal_estimate(t)
        # self.log_likilihood = np.log(self.marginal_likelihood)
        print("p:{},q:{},c0:{},c1:{},F:{},a0:{}\n LogLikelihood:{}".format(p,q,tr[2],tr[3],beta,self.a0,np.round(-np.sum(self.marginal_loglikelihood),5)))
        if __name__ == "__main__":
            return {"pro":self.marginal_probability,"a":self.a_marginal,"trance":self.TRANCE} # T*M :Pr(S_t|omega_t)
        else:
            return np.round(-np.sum(self.marginal_loglikelihood),5)
        # return self.eta

    def par_opt(self):
        # pp=np.random.uniform(0,5)
        # qq=np.random.uniform(0,5)
        pp = 1
        qq = 1
        c0 = 30
        c1 = 60
        beta = 0.6
        a00= 30
        a01= 60
        tr = np.array([pp,qq,c0,c1,beta,a00,a01])
        RESULT = optimize.minimize(self.model,tr)
        # RESULT = optimize.basinhopping(self.model,tr)
        return RESULT

if __name__ == '__main__':
    print("hello world")
