import symbolica_community
from symbolica_community import Expression, S, E
from symbolica_community.tensors import TensorLibrary, TensorNetwork, Representation, TensorStructure, TensorIndices, Tensor, Slot
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

dim = S("dim")


def curate(expr: Expression) -> Expression:
    expr = expr.replace(E_sp("Metric(x_,y_)"), E_sp("g(x_,y_)"), repeat=True)
    expr = expr.replace(E_sp("id(x_,y_)"), E_sp("𝟙(x_,y_)"), repeat=True)
    expr = expr.replace(E("spenso::γ(x_,y_,z_)"), E(
        "alg::gamma(x_,y_,z_)"), repeat=True)
    expr = expr.replace(E("spenso::T(x_,y_,z_)"),
                        E("alg::t(x_,y_,z_)"), repeat=True)
    expr = expr.replace(E("spenso::f(x_,y_,z_)"),
                        E("alg::f(x_,y_,z_)"), repeat=True)
    expr = expr.replace(E("spenso::TR"), E("alg::TR"), repeat=True)
    expr = expr.replace(E("spenso::Nc"), E("alg::Nc"), repeat=True)
    expr = expr.replace(E("spenso::v(x__)"), E("alg::v(x__)"), repeat=True)
    expr = expr.replace(E("spenso::vbar(x__)"),
                        E("alg::vbar(x__)"), repeat=True)
    expr = expr.replace(E("spenso::u(x__)"), E("alg::u(x__)"), repeat=True)
    expr = expr.replace(E("spenso::ubar(x__)"),
                        E("alg::ubar(x__)"), repeat=True)
    expr = expr.replace(E("spenso::ϵ(x__)"), E("alg::ϵ(x__)"), repeat=True)
    expr = expr.replace(E("spenso::ϵbar(x__)"),
                        E("alg::ϵbar(x__)"), repeat=True)
    expr = expr.replace(E("spenso::mink(4,x_)"), E(
        "spenso::mink(python::dim,x_)"), repeat=True)
    expr = expr.replace(E("spenso::coad(8,x_)"), E(
        "spenso::coad(alg::Nc^2-1,x_)"), repeat=True)
    expr = expr.replace(E("spenso::cof(3,x_)"), E(
        "spenso::cof(alg::Nc,x_)"), repeat=True)

    return expr


graph_one = E_sp(input[0]['expression'])
graph_one = curate(graph_one)
print("INPUT EXPR: ", graph_one)
expr = graph_one


mink = Representation.mink(dim)
Q = TensorStructure(mink, name=S("spenso::Q"))


def q(i, j):
    return Q(i, ';', j)


gamma = TensorStructure.gammadD(dim)
g = TensorStructure.metric(mink)


bis = Representation.bis(4)
u, ubar, v, vbar, eps, epsbar = S(
    "alg::u", "alg::ubar", "alg::v", "alg::vbar", "alg::ϵ", "alg::ϵbar")
i_, j_, d_, a_, b_ = S("i_", "j_", "d_", "a_", "b_")
dummy = S("dummy")
left = S("l")
right = S("r")


def square_sum(expr: Expression) -> Expression:
    lefta = wrap_dummies(expr, left)
    righta = conj(wrap_dummies(expr, right))
    square = (lefta*righta).expand()
    square = square.replace(eps(i_, mink(
        left(a_)))*epsbar(i_, mink(right(a_))), -g(left(a_), right(a_)), repeat=True)
    square = square.replace(eps(i_, mink(
        right(a_)))*epsbar(i_, mink(left(a_))), -g(left(a_), right(a_)), repeat=True)
    square = square.replace(vbar(i_, bis(left(a_)))*v(i_, bis(right(a_))), -gamma(
        dummy(i_, a_), left(a_), right(a_))*q(i_, dummy(i_, a_)), repeat=True)
    square = square.replace(vbar(i_, bis(right(a_)))*v(j_, bis(left(a_))), -gamma(
        dummy(i_, a_), left(a_), right(a_))*q(i_, dummy(i_, a_)), repeat=True)
    square = square.replace(ubar(i_, bis(left(a_)))*u(j_, bis(right(a_))), gamma(
        dummy(i_, a_), right(a_), left(a_))*q(i_, dummy(i_, a_)), repeat=True)
    square = square.replace(ubar(i_, bis(right(a_)))*u(j_, bis(left(a_))), gamma(
        dummy(i_, a_), right(a_), left(a_))*q(i_, dummy(i_, a_)), repeat=True)

    return square


square = square_sum(expr)
print("\n-->\nSQUARED EXPR", square)

# a = to_dots(simplify_color(simplify_gamma(simplify_gamma(square))))
# b = to_dots(simplify_gamma(simplify_color(square)))
# print((a-b).expand())
# a = to_dots(simplify_color(simplify_gamma(simplify_metrics(square))))
# b = to_dots(simplify_gamma(simplify_color(square)))
# print((a-b).expand())
# stop

# square = cook_indices(square)
square = simplify_color(square)
# square = simplify_gamma(square)
# square =simplify_metrics(expand_mink(square))
print("\n-->\nSQUARED EXPB", square)

# square = to_dots(simplify_gamma(square)).expand()
# square = simplify_color(square)

# square = to_dots(square)
lib = TensorLibrary.weyl()

square=square.replace(E("alg::gamma(x_,y_,z_)"),
                    E("weyl::gamma(x_,y_,z_)"), repeat=True)
net = TensorNetwork.from_expression(square.factor(),lib)

print(net)

net.execute(lib)

# res = net.result_scalar()

# nu = Slot("mink",4,"nu")
# nue = nu.to_expression()



# params = [Expression.I]
# params += TensorNetwork(Q(0,nue)).result_tensor(lib)
# params += TensorNetwork(Q(1,nue)).result_tensor(lib)
# params += TensorNetwork(Q(2,nue)).result_tensor(lib)
# params += TensorNetwork(Q(3,nue)).result_tensor(lib)
# constants = {S("alg::G"): E("1"),S("alg::Nc"): E("1"),S("alg::TR"): E("1")}

# # Much like the expressions, tensors have the same evaluation api, just that they return a tensor instead of an expression
# e=res.evaluator(constants=constants, params=params, funs={})
# # The evaluator can be compiled to a shared library
# c = e.compile(function_name="f", filename="test_expression.cpp",
#               library_name="test_expression.so", inline_asm=False)


# e_params = [random.random()+1j*random.random() for i in range(len(params))]
# eval_res = e.evaluate_complex([e_params])[0]

# print(eval_res)


#
# print("\n-->\nGAMMA SIMPLIFIED EXPR", square.factor())


# c1 = E("alg::t(spenso::coad(8,6),spenso::cof(3,5),spenso::dind(spenso::cof(3,4)))")

# c2 = E("alg::t(spenso::coad(8,6),spenso::cof(3,4),spenso::dind(spenso::cof(3,5)))")

# l = E("a+b+c")
# l2 = E("c+g")

# res = simplify_color(c1*l*c2*l2)
# print(res)




# def curate_two(expr: Expression) -> Expression:
#     expr = expr.replace(E_sp("Metric(x_,y_)"), E_sp("g(x_,y_)"), repeat=True)
#     expr= expr.replace(E_sp("id(cof(x__),y_)"),E_sp("1"),repeat=True)
#     expr = expr.replace(E_sp("id(x_,y_)"), E_sp("𝟙(x_,y_)"), repeat=True)
#     expr = expr.replace( E_sp("𝟙(cof(x__),y_)"),E_sp("1"), repeat=True)
#     expr= expr.replace(E_sp("𝟙(coad(x__),y_)"),E_sp("1"),repeat=True)
#     expr = expr.replace(E("spenso::γ(x_,y_,z_)"), E(
#         "weyl::gamma(x_,y_,z_)"), repeat=True)
#     expr = expr.replace(E("spenso::T(x_,y_,z_)"),
#                         E("1"), repeat=True)
#     expr = expr.replace(E("spenso::f(x_,y_,z_)"),
#                         E("1"), repeat=True)
#     expr = expr.replace(E("spenso::TR"), E("alg::TR"), repeat=True)
#     expr = expr.replace(E("spenso::Nc"), E("alg::Nc"), repeat=True)
#     expr = expr.replace(E("spenso::v(x__)"), E("alg::v(x__)"), repeat=True)
#     expr = expr.replace(E("spenso::vbar(x__)"),
#                         E("alg::vbar(x__)"), repeat=True)
#     expr = expr.replace(E("spenso::u(x__)"), E("alg::u(x__)"), repeat=True)
#     expr = expr.replace(E("spenso::ubar(x__)"),
#                         E("alg::ubar(x__)"), repeat=True)
#     expr = expr.replace(E("spenso::ϵ(x__)"), E("alg::ϵ(x__)"), repeat=True)
#     expr = expr.replace(E("spenso::ϵbar(x__)"),
#                         E("alg::ϵbar(x__)"), repeat=True)
#     # expr = expr.replace(E("spenso::mink(4,x_)"), E(
#         # "spenso::mink(4,x_)"), repeat=True)
#     expr = expr.replace(E("spenso::coad(8,x_)"), E(
#         "spenso::coad(alg::Nc^2-1,x_)"), repeat=True)
#     expr = expr.replace(E("spenso::cof(3,x_)"), E(
#         "spenso::cof(alg::Nc,x_)"), repeat=True)

#     return expr

# graph_one = curate_two(E_sp(input[0]['expression'])
# )
# print(graph_one)
# net = TensorNetwork.from_expression(graph_one,lib)
# print(net)

# net.execute(lib)
