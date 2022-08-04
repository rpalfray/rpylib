"""Testing the COS method"""

import pytest

from rpylib.model.utils import *
from rpylib.numerical.cosmethod import COSPricer


@pytest.mark.parametrize(
    'y, expected', [
        (0.5, 19.812948843),
        (1.5, 49.790905469),
        (1.98, 99.999905510),
    ]
)
def test_cos_method_cgmy_from_fang_osterlee(y, expected):
    n = 2**8
    spot = 100.0
    model = create_exponential_of_levy_model(ModelType.CGMY)(spot=spot, r=0.1, d=0.0, c=1.0, g=5.0, m=5.0, y=y)
    cos_pricer = COSPricer(model, n)

    strike = 100
    maturity = 1.0
    result = cos_pricer.call(strike, maturity)

    assert(np.isclose(result, expected, rtol=1e-10, atol=1e-9))


@pytest.mark.parametrize(
    'maturity, expected', [
        (0.1, 10.993703187),
        (1.0, 19.099354724),
    ]
)
def test_cos_method_variance_gamma_from_fang_osterlee(maturity, expected):
    n = 2**12
    spot = 100.0

    model = create_exponential_of_levy_model(ModelType.VG)(spot=spot, r=0.1, d=0., nu=0.2, sigma=0.12, theta=-0.14)
    cos_pricer = COSPricer(model, n)

    strike = 90
    result = cos_pricer.call(strikes=strike, time=maturity)

    assert(np.isclose(result, expected, rtol=1e-9, atol=1e-9))


def test_cos_method_consistency_variance_gamma_cgmy():
    n = 2**12
    spot, r, d = 100.0, 0.1, 0.
    nu, sigma, theta = 0.2, 0.12, -0.14
    eta_p = np.sqrt(theta**2*nu**2/4 + sigma**2*nu/2) + theta*nu/2
    eta_m = eta_p - theta*nu
    vg_model = create_exponential_of_levy_model(ModelType.VG)(spot=spot, r=r, d=d, nu=nu, sigma=sigma, theta=theta)
    cgmy_model = create_exponential_of_levy_model(ModelType.CGMY)(spot=spot, r=r, d=d, c=1/nu, g=1/eta_m, m=1/eta_p,
                                                                  y=0)

    vg_cos_pricer = COSPricer(vg_model, n)
    cgmy_cos_pricer = COSPricer(cgmy_model, n)

    maturity = 1.3
    strikes = 10.0 + np.arange(50+1)*180/50
    result_vg = vg_cos_pricer.call(strikes, maturity)
    result_cgmy = cgmy_cos_pricer.call(strikes, maturity)
    assert(all(np.isclose(result_vg, result_cgmy, rtol=0, atol=1e-8)))


def test_cos_method_bs_calls():
    spot = 100.0
    r = 0.05
    d = 0.02
    sigma = 0.30
    maturity = 7.0
    bs_model = create_exponential_of_levy_model(ModelType.BLACKSCHOLES)(spot=spot, r=r, d=d, sigma=sigma)
    bs_cos_pricer = COSPricer(bs_model)
    bs_cf_pricer = bs_model.closed_form
    
    strikes = 10.0 + np.arange(50+1)*180/50
    cos_prices = bs_cos_pricer.call(strikes, maturity)
    cf_prices = bs_cf_pricer.call(strikes, maturity)
    diffs = [abs(x-exp) for x, exp in zip(cos_prices, cf_prices)]

    assert(all(diff < 1e-13 for diff in diffs))


def test_cos_method_bs_digital():
    spot = 100.0
    r = 0.05
    d = 0.00
    sigma = 0.15
    maturity = 0.25
    bs_model = create_exponential_of_levy_model(ModelType.BLACKSCHOLES)(spot=spot, r=r, d=d, sigma=sigma)
    bs_cos_pricer = COSPricer(bs_model, n=500, l=20)
    bs_cf_pricer = bs_model.closed_form

    strikes = 10.0 + np.arange(50+1)*180/50
    cos_prices = bs_cos_pricer.digital(strikes, maturity)
    cf_prices = bs_cf_pricer.digital(strikes, maturity)
    diffs = [abs(x-exp) for x, exp in zip(cos_prices, cf_prices)]

    assert(all(diff < 1e-13 for diff in diffs))
