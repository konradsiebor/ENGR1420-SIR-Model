import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 1.25E7
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 100, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta; mean recovery rate, gamma, (in 1/days); death rate
beta, gamma, death_rate = 0.25, 1/12, 0.034
# A grid of time points (in days)
t = np.linspace(0, 800, 800)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -(beta * S * I) / N
    dIdt = (beta * S * I) / N - (gamma * I)
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Note to Zack: don't do parameter passing like this ever again, this is just one special case where it kinda works
def plot_and_save(t, S, I, R, D, N, i, title, beta):
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered')    
    ax.plot(t, D/N, '#000000', alpha=0.5, lw=2, label='Dead')

    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,1.2)
    ax.set_title(title+"; r0 = {:.2f}".format(beta))
    ax.annotate("Total deaths: {:.0f}".format(D[-1]), (110,.4))
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.savefig("SIR_graphs/reg_pop_{:02d}".format(i))

mitigation_pop = (0, .2, .3, .4, .5, .6)
beta_pop = [beta-beta*mit for mit in mitigation_pop]
mitigation_at_risk = (0, .05, .4, .5, .6, .7)
beta_risk = [beta-beta*mit for mit in mitigation_pop]

for i, pop in enumerate(beta_pop):
    # Initial conditions vector
    y0 = (S0, I0, R0)
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, pop, gamma))
    S, I, R = ret.T
    D = R*death_rate
    R = R-D
    plot_and_save(t, S, I, R, D, N, i, 'General population SIR model for COVID-19', pop)

death_rate = 0.2
for i, pop in enumerate(beta_risk):
    # Initial conditions vector
    y0 = (S0, I0, R0)
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, pop, gamma))
    S, I, R = ret.T
    D = R*death_rate
    R = R-D
    plot_and_save(t, S, I, R, D, N, i+6, 'At-Risk population SIR model for COVID-19', pop)



