from symbolica_community import Expression, S,E
from symbolica_community.tensors import TensorNetwork,Representation,TensorStructure,TensorIndices,Tensor,Slot,TensorLibrary
import symbolica_community
import symbolica_community.tensors as tensors
import random


# Tensor structures and representations:

# The starting point for defining a tensor is to define its structure. This is done by defining the indices that the tensor has. Each index is in fact a slot, built from an index and a representation
# Representations are defined by their name and dimension. They can be self-dual or not.
mink = Representation("mink",4)
bis = Representation("bis",4)
lor = Representation("lor",4,is_self_dual=False)

# Slots are created from a representation and an index
mu = mink("mu")
print(mu)
# They can be converted to symbolica expressions
mue = mu.to_expression()
print(mue)

# They can be built independently as well:
nu = Slot("mink",4,"nu")
nue = nu.to_expression()

# The index can be a string (that could be parsed into a symbolica symbol)
i = bis("i")
ie = i.to_expression()
# The index can also be an integer
j = bis(2)
je = j.to_expression()

k = S("k")
# The index can also directly a symbolica expression
k = bis(k)
ke = k.to_expression()


gamma,p,w,mq,id = S("weyl::gamma","P","W","mq","spenso::𝟙")

# Tensor structures are essentially a list of slots, with a name (symbolica symbol)
# It can be turned into a symbolica expression

g_muik = TensorIndices(gamma,mu,i,k)

print(g_muik)
print(g_muik[2])
print(g_muik[45:63:3])
print(g_muik[[2,2,2]])

# Spenso can then turn an expression that uses these slots into a tensor network
x = g_muik.to_expression()*(p(2,nue)*gamma(nue,ke,je)+mq*id(ke,je))*w(1,ie)*w(3,mue)
lib =TensorLibrary.weyl()
tn = TensorNetwork.from_expression(x,lib)
# prints the rich graph associated to the network
print(tn)
# As you can see when parsed, the network isn't contracted yet, so it's just a graph of the expression
# To contract it, you can call the contract method
tn.execute(lib)
# The graph is now a single node (or at least has no internal edges)
print(tn)
# We can now extract the resulting tensor:
t = tn.result_tensor(lib)
# The tensor is a tensor object
print(t)
# It has a structure
print(t.structure())

# Evaluation of the tensor
#
# You may have noticed that the resulting tensor is a set of expressions with certain functions that label the 'concrete' values of the tensor. What if we want to evaluate the tensor for a given set of parameters?

params = [Expression.I]
params += TensorNetwork(w(1,ie)).result_tensor(lib)# tensors implement the sequence protocol, so can be treated just like lists
params += TensorNetwork(w(3,mue)).result_tensor(lib)
params += TensorNetwork(p(2,nue)).result_tensor(lib)
constants = {mq: E("173")}

# Much like the expressions, tensors have the same evaluation api, just that they return a tensor instead of an expression
e=t.evaluator(constants=constants, params=params, funs={})
# The evaluator can be compiled to a shared library
c = e.compile(function_name="f", filename="test_expression.cpp",
              library_name="test_expression.so", inline_asm=False)


e_params = [random.random()+1j*random.random() for i in range(len(params))]
eval_res = e.evaluate_complex([e_params])[0]

print(eval_res)
print(eval_res.structure())

# Tensor building:

# Tensors have two storage modes. Hashmap backed sparse tensors and dense tensors. Sparse tensors are built from a structure and then assigned values:

t = tensors.sparse_empty([lor,lor],type(gamma))
# Note that the structure is a list of representations, not slots
# In this case the tensor structure is a `TensorStructure` object
print(t.structure())
# It does not have indices, just a shape. This makes sense if you just want to register this tensor (see later), or if you just want to use it for storage and iteration.
# The tensor can store expressions,floats or complex numbers (homogeneously)
# values are accessed by flattened index: (and slices are supported)
t[6]=E("f(x)*(1+y)")
# Or by multi-index:
t[[3,2]]=E("sin(alpha)")
print(t)
# Currently printing makes more sense when the data is dense
t.to_dense()
print(t)

d = Representation("newrep",3)
tname = S("test")
#  Dense tensors are built from a list of values in row-major order.
t = tensors.dense(TensorStructure(d,d,name=tname),[0,0,123,
                        11,3,234,
                        234,23,44,])
# If the structure is just a list of integers, it is assumed to be a list of dimensions, and the representation is assumed to be the default representation: euclidean.
t[[1,2]]=3/34
print(t)
print(t.structure())


lib.register(t)

def new_t(i,j):
    return TensorIndices(tname,d(i),d(j))


x=(new_t(1,2)*new_t(2,3)*new_t(3,1))
n =  TensorNetwork.from_expression(x,lib)
n.execute(lib)
t = n.result_tensor(lib)
print(t)
