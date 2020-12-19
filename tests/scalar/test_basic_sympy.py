import pytest

import theano


sympy = pytest.importorskip("sympy")

from theano.scalar.basic import floats
from theano.scalar.basic_sympy import SymPyCCode


xs = sympy.Symbol("x")
ys = sympy.Symbol("y")

xt, yt = floats("xy")


@pytest.mark.skipif(not theano.config.cxx, reason="Need cxx for this test")
def test_SymPyCCode():
    op = SymPyCCode([xs, ys], xs + ys)
    e = op(xt, yt)
    g = theano.gof.FunctionGraph([xt, yt], [e])
    fn = theano.link.c.cc.CLinker().accept(g).make_function()
    assert fn(1.0, 2.0) == 3.0


def test_grad():
    op = SymPyCCode([xs], xs ** 2)
    zt = op(xt)
    ztprime = theano.grad(zt, xt)
    assert ztprime.owner.op.expr == 2 * xs


def test_multivar_grad():
    op = SymPyCCode([xs, ys], xs ** 2 + ys ** 3)
    zt = op(xt, yt)
    dzdx, dzdy = theano.grad(zt, [xt, yt])
    assert dzdx.owner.op.expr == 2 * xs
    assert dzdy.owner.op.expr == 3 * ys ** 2
