"""Testing the FFT pricing method"""

import pytest

from rpylib.model.utils import *
from rpylib.numerical.fft import FFTPricer


def compute_fft_otm_prices(model, strikes, maturity):
    fft_prices = []
    fft_pricer = FFTPricer(model)

    for strike in strikes:
        if strike < model.spot:
            price = fft_pricer.put(strike, maturity)
        else:
            price = fft_pricer.call(strike, maturity)
        fft_prices.append(price)
        
    return fft_prices


def compute_cf_otm_prices(model, strikes, maturity):
    cf_prices = []
    for strike in strikes:
        if strike < model.spot:
            price = model.closed_form.put(strike, maturity)
        else:
            price = model.closed_form.call(strike, maturity)
        cf_prices.append(price)
        
    return cf_prices


def bs_model():
    model = create_exponential_of_levy_model(ModelType.BLACKSCHOLES)(spot=100., r=0.03, d=0.01, sigma=0.30)
    return model


@pytest.fixture(scope='module')
def cgmy_model():
    # from the paper *** by Madan and Yor
    model = create_exponential_of_levy_model(ModelType.CGMY)(spot=100., r=0.03, d=0.01, c=1.0, g=5.0, m=10.0, y=0.5)
    return model


@pytest.fixture(scope='module')
def cgmy_model_II():
    # from the paper 'Robust  numerical  valuation  of  European  and  American  options  under  the  CGMY  process'
    # by Forsyth, P. A., J. W. Wan, and I. R. Wang r
    model = create_exponential_of_levy_model(ModelType.CGMY)(spot=90., r=0.06, d=0.00, c=16.97, g=7.08, m=29.97,
                                                             y=0.6442)
    return model


@pytest.fixture(scope='module')
def cgmy_model_III():
    # from the paper 'Robust  numerical  valuation  of  European  and  American  options  under  the  CGMY  process'
    # by Forsyth, P. A., J. W. Wan, and I. R. Wang r
    model = create_exponential_of_levy_model(ModelType.CGMY)(spot=90., r=0.06, d=0.00, c=0.42, g=4.37, m=191.2,
                                                             y=1.0102)
    return model


@pytest.mark.slow
def test_fft_blackscholes():
    threshold = 1e-6
    bs_model_instance = bs_model()
    maturity = 1.5
    strikes = 60.0 + np.arange(10+1)*80/10
    fft_prices = compute_fft_otm_prices(bs_model_instance, strikes, maturity)
    cf_prices = compute_cf_otm_prices(bs_model_instance, strikes, maturity)
    assert(all(abs(p1 - p0) < threshold for p1, p0 in zip(fft_prices, cf_prices)))


@pytest.mark.slow
@pytest.mark.parametrize(
    'maturity, strikes, benchmark', [
        (0.25, [80.0, 90.0, 100.0, 110.0, 120.0], [0.8698, 2.2475, 5.8919, 2.1420, 0.7848]),
        (0.50, [80.0, 90.0, 100.0, 110.0, 120.0], [1.8854, 4.0589, 8.8226, 4.7026, 2.3585]),
        (0.75, [80.0, 90.0, 100.0, 110.0, 120.0], [2.8638, 5.5219, 11.0672, 6.8875, 4.1038]),
        (1.00, [80.0, 90.0, 100.0, 110.0, 120.0], [3.7681, 6.7515, 12.9545, 8.7819, 5.7803])
    ], ids=[
        'T=0.25',
        'T=0.50',
        'T=0.75',
        'T=1.00',
    ]
)
def test_fft_cgmy(cgmy_model, maturity, benchmark, strikes):
    threshold = 0.005
    fft_prices = compute_fft_otm_prices(cgmy_model, strikes, maturity)
    assert(all(abs(fft_price - cf_price) < threshold for fft_price, cf_price in zip(fft_prices, benchmark)))


def test_fft_cgmy_e(cgmy_model_II):
    maturity = 0.25
    strikes = [98.0]
    benchmark = [16.212375]
    cgmy_ffft_pricer = FFTPricer(cgmy_model_II)
    fft_call = cgmy_ffft_pricer.call(strikes, maturity)
    assert(abs(fft_call[0] - benchmark) < 1e-3)


def test_fft_cgmy_f(cgmy_model_III):
    maturity = 0.25
    strikes = [98.0]
    benchmark = 2.2306913
    cgmy_ffft_pricer = FFTPricer(cgmy_model_III)
    fft_call = cgmy_ffft_pricer.call(strikes, maturity)
    assert(abs(fft_call[0] - benchmark) < 1e-4)
