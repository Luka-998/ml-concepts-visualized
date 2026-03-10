import numpy as np
import matplotlib.pyplot as plt


class Cost():
    def __init__(self,x,y,w,b,):
        self.x=x
        self.y=y
        self.w=w
        self.b=b
    def get_cost_loop(self):
        m = self.x.shape[0]
        n = self.x.shape[1]
        #print(self.x.shape)
        #print(self.w.shape)
        total_cost = 0
        for i in range(m): # MATRIX MULTIPLICATION REMINDER! -> if X is matrix in shape of (3,4), by taking all the rows of 1st column * row vector W = correct way also 
            pred = np.dot(self.w,self.x[i]) + self.b
            error = pred - self.y[i]
            error_sq = error**2
            total_cost +=error_sq
            cost = 1/(2*m)*total_cost.sum()

        return cost

    def get_cost_no_loop(self):
        m = self.x.shape[0]
        pred = np.dot(self.x,self.w)+self.b
        error = (self.y-pred)**2
        cost = 1/(2*m)*error.sum()
        return cost
    

class Round1():
    def __init__(self,x,y,w,b):
        self.x=x
        self.y=y
        self.w=w
        self.b=b
        self.m = len(w)
        self.n = len(x)
    def meshgrid(self):
        B,W = np.meshgrid(self.b,self.w)
        return B,W
    def pre_calculation(self):
        """ Preparing the data for arithemtic computation"""
        B,W = self.meshgrid()

        # Dynamic reshape here:
        B_3d = B.reshape(self.m,self.m,1)
        W_3d = W.reshape(self.m,self.m,1)
        X_3d = self.x.reshape(1,1,self.n)
        Y_3d = self.y.reshape(1,1,self.n)
        return B_3d,W_3d,X_3d,Y_3d
    
    def get_cost(self):
        b,w,x,y = self.pre_calculation()
        pred = w*x + b
        residuals = (y-pred)**2
        cost = 1/(2*self.n)*np.sum(residuals**2,axis=2)
        return cost
    def plot3d(self):
        B,W = self.meshgrid()
        cost_values = self.get_cost()
        fig= plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_surface(B,W,cost_values)
        ax.view_init(25,25)
        plt.show()

