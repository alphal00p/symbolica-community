from symbolica_community import Expression, S, E
from symbolica_community.tensors import TensorNetwork, Representation, TensorStructure, TensorIndices, Tensor, Slot
import symbolica_community
import symbolica_community.tensors as tensors
import random
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


def curate(expr: Expression) -> Expression:
    expr = expr.replace(E_sp("Metric(x_,y_)"), E_sp("g(x_,y_)"))
    expr = expr.replace(E_sp("id(x_,y_)"), E_sp("𝟙(x_,y_)"))
    expr = expr.replace(E("spenso::γ(x_,y_,z_)"), E("alg::gamma(x_,y_,z_)"))
    expr = expr.replace(E("spenso::T(x_,y_,z_)"), E("alg::t(x_,y_,z_)"))
    expr = expr.replace(E("spenso::f(x_,y_,z_)"), E("alg::f(x_,y_,z_)"))
    expr = expr.replace(E("spenso::TR"), E("alg::TR"))
    expr = expr.replace(E("spenso::Nc"), E("alg::Nc"))
    return expr


graph_one = E_sp(input[0]['expression'])
graph_one = curate(graph_one)
print("INPUT EXPR: ", printer(graph_one))
# print(simplify_color(curate(graph_one))
expr = graph_one

expr = simplify_gamma(expr)
print("\n-->\nGAMMA SIMPLIFIED EXPR", printer(expr))

expr = simplify_color(graph_one)
print("\n-->\nCOLOR SIMPLIFIED EXPR", printer(expr))
