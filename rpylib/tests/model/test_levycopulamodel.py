import numpy as np

from rpylib.distribution.levycopula import ClaytonCopula
from rpylib.model.levycopulamodel import LevyCopulaModel
from rpylib.model.levymodel.mixed.hem import HEMParameters, ExponentialOfHEMModel
from rpylib.model.levymodel.purejump.cgmy import CGMYParameters, ExponentialOfCGMYModel


def test_levycopulamodel2d_volume():

    spot1, spot2, r, d1, d2 = 100.0, 80.0, 0.02, 0.0, 0.0
    c1, g1, m1, y1 = 0.1, 20.0, 8.0, 0.5
    c2, g2, m2, y2 = 0.02, 10.0, 8.0, 0.9
    model1 = ExponentialOfCGMYModel(
        spot=spot1, r=r, d=d1, parameters=CGMYParameters(c=c1, g=g1, m=m1, y=y1)
    )
    model2 = ExponentialOfCGMYModel(
        spot=spot2, r=r, d=d2, parameters=CGMYParameters(c=c2, g=g2, m=m2, y=y2)
    )
    theta, eta = 0.7, 0.3
    copula = ClaytonCopula(theta=theta, eta=eta)
    levy_copula_model = LevyCopulaModel(models=[model1, model2], copula=copula)

    cases = [
        {"a": [0.05, 0.03], "b": [0.1, 0.08]},
        {"a": [-0.10, 0.03], "b": [-0.05, 0.08]},
        {"a": [0.05, -0.08], "b": [0.1, -0.03]},
        {"a": [-0.10, -0.08], "b": [-0.05, -0.03]},
        {"a": [-0.05, -0.08], "b": [0.05, -0.03]},
        {"a": [0.01, -0.05], "b": [0.05, 0.05]},
    ]

    ts = [levy_copula_model._mass_nd(case["a"], case["b"]) for case in cases]
    tts = [levy_copula_model._mass_2d(case["a"], case["b"]) for case in cases]

    for t, tt in zip(ts, tts):
        assert np.isclose(t, tt)


def test_levycopulamodel3d_volume():

    spot1, spot2, spot3, r, d1, d2, d3 = 100.0, 80.0, 130, 0.02, 0.0, 0.0, 0.0
    c1, g1, m1, y1 = 0.1, 20.0, 8.0, 0.5
    c2, g2, m2, y2 = 0.02, 10.0, 8.0, 0.9
    sigma, p, eta1, eta2, intensity = 0.10, 0.6, 20, 25, 3
    model1 = ExponentialOfCGMYModel(
        spot=spot1, r=r, d=d1, parameters=CGMYParameters(c=c1, g=g1, m=m1, y=y1)
    )
    model2 = ExponentialOfCGMYModel(
        spot=spot2, r=r, d=d2, parameters=CGMYParameters(c=c2, g=g2, m=m2, y=y2)
    )
    model3 = ExponentialOfHEMModel(
        spot=spot3,
        r=r,
        d=d3,
        parameters=HEMParameters(
            sigma=sigma, p=p, eta1=eta1, eta2=eta2, intensity=intensity
        ),
    )
    theta, eta = 0.7, 0.3
    copula = ClaytonCopula(theta=theta, eta=eta)
    levy_copula_model = LevyCopulaModel(models=[model1, model2, model3], copula=copula)

    cases = [
        {
            "a": [+0.05, +0.03, +0.04],
            "b": [+0.1, +0.08, +0.06],
        },  # 00 < a1 < b1,   0 < a2 < b2,   0 < a3 < b3
        {
            "a": [-0.05, +0.03, +0.04],
            "b": [+0.1, +0.08, +0.06],
        },  # a1 <  0 < b1,   0 < a2 < b2,   0 < a3 < b3
        {
            "a": [-0.05, -0.03, +0.04],
            "b": [+0.1, +0.08, +0.06],
        },  # a1 <  0 < b1,  a2 <  0 < b2,   0 < a3 < b3
        {
            "a": [-0.05, +0.03, -0.04],
            "b": [+0.1, +0.08, +0.06],
        },  # a1 <  0 < b1,   0 < a2 < b2,  a3 <  0 < b3
        {
            "a": [+0.05, -0.03, -0.04],
            "b": [+0.1, +0.08, +0.06],
        },  # 00 < a1 < b1,  a2 <  0 < b2,  a3 <  0 < b3
        {
            "a": [-0.05, -0.03, -0.04],
            "b": [-0.1, -0.08, -0.06],
        },  # a1 < b1 <  0,  a2 < b2 <  0,  a3 < b3 < 0
        {
            "a": [-0.05, -0.03, -0.04],
            "b": [+0.1, -0.08, -0.06],
        },  # a1 <  0 < b1,  a2 < b2 <  0,  a3 < b3 < 0
        {
            "a": [-0.05, -0.03, -0.04],
            "b": [-0.1, +0.08, -0.06],
        },  # a1 < b1 <  0,  a2 <  0 < b2,  a3 < b3 < 0
        {
            "a": [-0.05, -0.03, -0.04],
            "b": [-0.1, -0.08, +0.06],
        },  # a1 < b1 <  0,  a2 < b2 <  0,  a3 <  0 < b3
    ]

    ts = [levy_copula_model._mass_nd(case["a"], case["b"]) for case in cases]
    tts = [levy_copula_model._mass_3d(case["a"], case["b"]) for case in cases]

    for t, tt in zip(ts, tts):
        assert np.isclose(t, tt)
