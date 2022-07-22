import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')


def variance_upper_bound(model, h, T):
    d = 1
    lm = model.levy_triplet.nu
    int1_val = lm.integrate(-np.inf, -h/2) + lm.integrate(h/2, np.inf)
    int2_val = lm.integrate_against_xx(-h, h)
    return 16*T*d*h*h*int1_val + 64*d*d*T*int2_val


def polynomial_fit(figaxis, xaxis, yaxis, levels, startIndex, curvelabel):
    xaxis_cut = xaxis[startIndex:]
    yaxis_cut = yaxis[startIndex:]
    fit = np.polyfit(xaxis_cut, yaxis_cut, deg=1)
    a, b = fit[0], fit[1]
    figaxis.plot(levels, a*xaxis + b, color='g', linewidth=0.5)
    print('fitted curve for {:12}: '.format(curvelabel) + '{a:0.3}x + {b:0.3}'.format(a=a, b=b))
    return a, b


def retrieve_results(resultsfile):
    with open(resultsfile, newline='') as csvfile:
        df = pd.read_csv(csvfile, sep=',')
        mlevell = df['meanlevell'].tolist()
        vlevell = df['varlevell'].tolist()
        consistencycheck = df['consistencycheck'].tolist()
        kurtosis = df['kurtosis'].tolist()
        ml = df['ml'].tolist()
        vl = df['vl'].tolist()
        cl = df['cl'].tolist()

    return mlevell, vlevell, consistencycheck, kurtosis, ml, vl, cl


def plot_graphs(model, sFitIndex, h0, mlevell, vlevell, consistencycheck, kurtosis, ml, vl, cl):
    bg = model.blumenthal_getoor_index()

    print('Blumentahl Getoor index = {:.4f}'.format(bg))
    print('expected convergence rate vl = {:.2f}'.format(2.0 - bg))
    print('expected convergence rate ml = {:.2f}'.format(1.0 - bg/2.0))
    print('expected convergence rate cl = {:.2f}'.format(bg))
    print('')

    fig, (axml, axvl, axkurt, axcc, axcl) = plt.subplots(1, 5, figsize=(30, 8))

    log2_mlevell = np.log2([abs(val) for val in mlevell])
    levels = list(range(len(log2_mlevell)))
    hl = [h0/2**l for l in levels]

    log_hl = np.log2(hl)
    log2_vlevell = np.log2(vlevell)
    log2_ml = np.log2([abs(mlval) for mlval in ml[1:]])  # the level 0 has value 0
    log2_vl = np.log2(vl[1:])  # the level 0 has value 0
    log2_cl = np.log2(cl)

    axml.plot(levels, log2_mlevell, color='royalblue', label="$P_l$", marker='x', linewidth=1.0)
    axml.plot(levels[1:], log2_ml, color='b', label="$P_l - P_{l-1}$", marker='*', linewidth=1.0)
    aa, bb = polynomial_fit(axml, log_hl[1:], log2_ml, levels[1:], sFitIndex, 'means')
    ml_convergence_rate = aa

    axvl.plot(levels, log2_vlevell, color='royalblue', label="$P_l$", marker='*', linewidth=1.0)
    axvl.plot(levels[1:], log2_vl, color='b', label="$P_l - P_{l-1}$", marker='*', linewidth=1.0)

    a1, b1 = polynomial_fit(axvl, log_hl[1:], log2_vl, levels[1:], sFitIndex, 'variances')
    new_convergence_rate = a1

    axkurt.plot(levels[1:], kurtosis[1:], color='b', label="kurtosis", marker='*', linewidth=1.0)
    axcc.plot(levels[1:], consistencycheck[1:], color='b', label="consistency check", marker='*', linewidth=1.0)
    axcl.plot(levels, log2_cl, color='b', label="cost l", marker='*', linewidth=1.0)
    a_cl, b_cl = polynomial_fit(axcl, log_hl[1:], log2_cl[1:], levels[1:], sFitIndex, 'cost cl')
    cl_convergence_rate = a_cl

    avgcost = sum(log2_cl[-4:])/len(log2_cl[-4:]) - sum([np.log2(h**(-bg)) for h in hl[-4:]])/len(hl[-4:])
    axcl.plot(levels, avgcost + np.array([np.log2(h**(-bg)) for h in hl]), color='r', label="th cost l (slope)",
              marker='*', linewidth=1.0)

    axml.set_ylabel('$\log_2$|mean|')
    axml.set_xlabel('level l')

    axvl.set_ylabel('$\log_2$ variance')
    axvl.set_xlabel('level l')

    axcc.set_ylabel('Consistency check')
    axcc.set_xlabel('level l')

    axkurt.set_ylabel('Kurtosis')
    axkurt.set_xlabel('level l')

    axcl.set_ylabel('$C_l$')
    axcl.set_xlabel('cost l')

    axml.legend()
    axvl.legend()
    axcc.legend()
    axkurt.legend()
    axcl.legend()

    fig.tight_layout()
    plt.show()

    return new_convergence_rate, cl_convergence_rate, ml_convergence_rate
