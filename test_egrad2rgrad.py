import autograd.numpy as np
#import numpy as np
from autograd import grad
from pymanopt.manifolds import FixedRankEmbedded

def f(X):
    res = np.trace(np.dot(X,np.transpose(X)))

    return res

def g(USV):
    U = USV[0]
    S = np.diag(USV[1])
    Vt = USV[2]

    X = np.dot(np.dot(U,S),Vt)

    return f(X)

m = 4
n = 2
k = 2

X = np.random.rand(m,n)
U,S,Vt = np.linalg.svd(X,full_matrices=False)

U=U[:,:k]
S=S[:k]
Vt=Vt[:k,:]

grad_f = grad(f)
grad_g = grad(g)

man = FixedRankEmbedded(m,n,k)

dU,dS,dVt = grad_g((U,S,Vt))

Up,M,Vp = man.egrad2rgrad((U,S,Vt),(dU,dS,dVt))
U_,S_,V_ = man.tangent2ambient((U,S,Vt),(Up,M,Vp))

tangent_grad = U_.dot(S_).dot(V_.T)


print
print
print 'X:'
print X
print
print 'U,S,Vt:'
print U,S,Vt
print
print 'f(X):'
print f(X)
print
print 'g(U,S,V):'
print g((U,S,Vt))
print
print 'grad_f(X):'
print grad_f(X)
print
print 'grad_g(U,S,V):'
print grad_g((U,S,Vt))
print
print 'representation of tangent grad -- Up,M,Vp:'
print  Up,M,Vp
print
print 'tangent grad in USV -- U_,np.diag(S_),V_.T:'
print U_,np.diag(S_),V_.T
print
print 'tangent_grad:'
print tangent_grad
print
print 'is tangent_grad == grad_f(X) ?'
print np.allclose(tangent_grad,grad_f(X))



