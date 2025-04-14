# from symbolica_community import Expression, S, E
# from symbolica_community.tensors import TensorNetwork, Representation, TensorStructure, TensorIndices, Tensor, Slot/
# import symbolica_community
# import symbolica_community.tensors as tensors
# import random
from symbolica_community.algebraic_simplification import *


def E_sp(s):
    return E(s, default_namespace="spenso")


input = {
    0: {'expression': '-G^2*(-Metric(mink(4,5),mink(4,6))*Q(2,mink(4,7))+Metric(mink(4,5),mink(4,6))*Q(3,mink(4,7))+Metric(mink(4,5),mink(4,7))*Q(2,mink(4,6))+Metric(mink(4,5),mink(4,7))*Q(4,mink(4,6))-Metric(mink(4,6),mink(4,7))*Q(3,mink(4,5))-Metric(mink(4,6),mink(4,7))*Q(4,mink(4,5)))*Metric(mink(4,4),mink(4,7))*T(coad(8,6),cof(3,5),dind(cof(3,4)))*f(coad(8,7),coad(8,8),coad(8,9))*id(bis(4,0),bis(4,5))*id(bis(4,1),bis(4,4))*id(mink(4,2),mink(4,5))*id(mink(4,3),mink(4,6))*id(coad(8,2),coad(8,7))*id(coad(8,3),coad(8,8))*id(coad(8,6),coad(8,9))*id(cof(3,0),dind(cof(3,5)))*id(cof(3,4),dind(cof(3,1)))*γ(mink(4,4),bis(4,5),bis(4,4))*vbar(1,bis(4,1))*u(0,bis(4,0))*ϵbar(2,mink(4,2))*ϵbar(3,mink(4,3))',
        'momenta': {'Q(0,x___)': 'P(0,x___)',
                      'Q(1,x___)': 'P(1,x___)',
                    'Q(2,x___)': 'P(2,x___)',
                    'Q(3,x___)': 'P(0,x___)+P(1,x___)-P(2,x___)',
                    'Q(4,x___)': 'P(0,x___)+P(1,x___)'}},
    1: {'expression': '-𝑖*G^2*T(coad(8,6),cof(3,5),dind(cof(3,4)))*T(coad(8,9),cof(3,8),dind(cof(3,7)))*id(bis(4,0),bis(4,5))*id(bis(4,1),bis(4,6))*id(mink(4,2),mink(4,4))*id(mink(4,3),mink(4,5))*id(coad(8,2),coad(8,6))*id(coad(8,3),coad(8,9))*id(cof(3,0),dind(cof(3,5)))*id(cof(3,4),dind(cof(3,8)))*id(cof(3,7),dind(cof(3,1)))*γ(mink(4,4),bis(4,5),bis(4,4))*γ(mink(4,5),bis(4,7),bis(4,6))*γ(mink(4,20),bis(4,4),bis(4,7))*vbar(1,bis(4,1))*u(0,bis(4,0))*ϵbar(2,mink(4,2))*ϵbar(3,mink(4,3))*Q(4,mink(4,20))',
        'momenta': {'Q(0,x___)': 'P(0,x___)',
                      'Q(1,x___)': 'P(1,x___)',
                    'Q(2,x___)': 'P(2,x___)',
                    'Q(3,x___)': 'P(0,x___)+P(1,x___)-P(2,x___)',
                    'Q(4,x___)': 'P(0,x___)-P(2,x___)'}},
    2: {'expression': '-𝑖*G^2*T(coad(8,6),cof(3,5),dind(cof(3,4)))*T(coad(8,9),cof(3,8),dind(cof(3,7)))*id(bis(4,0),bis(4,5))*id(bis(4,1),bis(4,6))*id(mink(4,2),mink(4,5))*id(mink(4,3),mink(4,4))*id(coad(8,2),coad(8,9))*id(coad(8,3),coad(8,6))*id(cof(3,0),dind(cof(3,5)))*id(cof(3,4),dind(cof(3,8)))*id(cof(3,7),dind(cof(3,1)))*γ(mink(4,4),bis(4,5),bis(4,4))*γ(mink(4,5),bis(4,7),bis(4,6))*γ(mink(4,20),bis(4,4),bis(4,7))*vbar(1,bis(4,1))*u(0,bis(4,0))*ϵbar(2,mink(4,2))*ϵbar(3,mink(4,3))*Q(4,mink(4,20))',
        'momenta': {'Q(0,x___)': 'P(0,x___)',
                      'Q(1,x___)': 'P(1,x___)',
                    'Q(2,x___)': 'P(2,x___)',
                    'Q(3,x___)': 'P(0,x___)+P(1,x___)-P(2,x___)',
                    'Q(4,x___)': '-P(1,x___)+P(2,x___)'}}
}

dim = S("dim")

def curate(expr: Expression) -> Expression:
    expr = expr.replace(E_sp("Metric(x_,y_)"), E_sp("g(x_,y_)"),repeat=True)
    expr = expr.replace(E_sp("id(x_,y_)"), E_sp("𝟙(x_,y_)"),repeat=True)
    expr = expr.replace(E("spenso::γ(x_,y_,z_)"), E("alg::gamma(x_,y_,z_)"),repeat=True)
    expr = expr.replace(E("spenso::T(x_,y_,z_)"), E("alg::t(x_,y_,z_)"),repeat=True)
    expr = expr.replace(E("spenso::f(x_,y_,z_)"), E("alg::f(x_,y_,z_)"),repeat=True)
    expr = expr.replace(E("spenso::TR"), E("alg::TR"),repeat=True)
    expr = expr.replace(E("spenso::Nc"), E("alg::Nc"),repeat=True)
    expr = expr.replace(E("spenso::v(x__)"), E("alg::v(x__)"),repeat=True)
    expr = expr.replace(E("spenso::vbar(x__)"), E("alg::vbar(x__)"),repeat=True)
    expr = expr.replace(E("spenso::u(x__)"), E("alg::u(x__)"),repeat=True)
    expr = expr.replace(E("spenso::ubar(x__)"), E("alg::ubar(x__)"),repeat=True)
    expr = expr.replace(E("spenso::ϵ(x__)"), E("alg::ϵ(x__)"),repeat=True)
    expr = expr.replace(E("spenso::ϵbar(x__)"), E("alg::ϵbar(x__)"),repeat=True)
    expr = expr.replace(E("spenso::mink(4,x_)"), E("spenso::mink(python::dim,x_)"),repeat=True)
    expr = expr.replace(E("spenso::coad(8,x_)"), E("spenso::coad(alg::Nc^2-1,x_)"),repeat=True)
    expr = expr.replace(E("spenso::cof(3,x_)"), E("spenso::cof(alg::Nc,x_)"),repeat=True)

    return expr


graph_one = E_sp(input[0]['expression'])
graph_one = curate(graph_one)
print("INPUT EXPR: ", graph_one)
expr = graph_one


mink = Representation.mink(dim);
Q = TensorStructure(mink,name=S("spenso::Q"))

def q(i,j):
    return Q(i,';',j)

gamma = TensorStructure.gammadD(dim)
g = TensorStructure.metric(mink);


bis = Representation.bis(4);
u, ubar,v,vbar,eps,epsbar = S("alg::u","alg::ubar","alg::v","alg::vbar","alg::ϵ","alg::ϵbar")
i_, j_,d_,a_,b_ = S("i_", "j_","d_","a_","b_")
dummy = S("dummy")
left = S("l");
right = S("r");
def square_sum(expr:Expression)->Expression:
    lefta = wrap_dummies(expr,left)
    righta = conj(wrap_dummies(expr,right))
    square = (lefta*righta).expand()
    square = square.replace(eps(i_,mink(left(a_)))*epsbar(i_,mink(right(a_))),-g(left(a_),right(a_)),repeat=True)
    square = square.replace(eps(i_,mink(right(a_)))*epsbar(i_,mink(left(a_))),-g(left(a_),right(a_)),repeat=True)
    square = square.replace(vbar(i_,bis(left(a_)))*v(i_,bis(right(a_))),-gamma(dummy(i_,a_),left(a_) ,right(a_))*q(i_,dummy(i_,a_)),repeat=True)
    square = square.replace(vbar(i_,bis(right(a_)))*v(j_,bis(left(a_))),-gamma(dummy(i_,a_),left(a_) ,right(a_))*q(i_,dummy(i_,a_)),repeat=True)
    square = square.replace(ubar(i_,bis(left(a_)))*u(j_,bis(right(a_))),gamma(dummy(i_,a_),right(a_) ,left(a_))*q(i_,dummy(i_,a_)),repeat=True)
    square = square.replace(ubar(i_,bis(right(a_)))*u(j_,bis(left(a_))),gamma(dummy(i_,a_),right(a_) ,left(a_))*q(i_,dummy(i_,a_)),repeat=True)

    return square




square = square_sum(expr)
print("\n-->\nSQUARED EXPR", square)

expr = cook_indices(square)
expr = to_dots(simplify_gamma(expr))

print("\n-->\nGAMMA SIMPLIFIED EXPR", expr.factor())
