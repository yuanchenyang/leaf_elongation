"""Input data format: first column with name location; other columns as samples
with name of the sames as colnames. Can input a vector of cutoff for tails
otherwise use the whole column for fitting.

Output parameters in a csv, and output the fitting curve and original data in
figures in a pdf.

Output rownames are sample names, and colnames are parameters, add l10_90
"""

import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate

from utils import *

def get_xy(df, col, row, x_col='all_data'):
    filtered = df[~df.iloc[:, col].isna()]
    return np.array(filtered[x_col])[:row], np.array(filtered.iloc[:, col])[:row]

def sigmoid(x, L ,x0, k, b):
    '''
    L is responsible for scaling the output range from [0,1] to [0,L]
    b adds bias to the output and changes its range from [0,L] to [b,L+b]
    k is responsible for scaling the input, which remains in (-inf,inf)
    x0 is the point in the middle of the Sigmoid, i.e. the point where Sigmoid
    should originally output the value 1/2 [since if x=x0, we get 1/(1+exp(0)) =
    1/2].
    '''
    return L / (1 + np.exp(-k*(x-x0))) + b

def sigmoid_tail(x, L, x0, k, b, x1, a1, a2):
    out = sigmoid(x, L, x0, k, b)
    tail = x[x > x1]
    out[x > x1] =  sigmoid(x1, L, x0, k, b) + a1 * (tail-x1) + a2 * (tail-x1)**2
    return out

def make_sigmoid_tail(L, x0, k, b, x1):
    def sigmoid_tail(x, a1, a2):
        out = sigmoid(x, L, x0, k, b)
        tail = x[x > x1]
        out[x > x1] = sigmoid(x[x > x1], L, x0, k, b) + a1 * (tail-x1) + a2 * (tail-x1)**2
        return out
    return sigmoid_tail

def get_rsquared(f, x, y, popt):
    residuals = y - f(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    return 1 - (ss_res / ss_tot)

def main():
    parser = get_default_parser()
    parser.add_argument('-e', '--ext',
                        help='Filename extension of input csv',
                        default='_in.csv')
    parser.add_argument('-n', '--new_ext',
                        help='Filename extension of output csv',
                        default='_params.csv')
    parser.add_argument('-p', '--plot_ext',
                        help='Filename extension of output plots',
                        default='_plot.png')
    args = parser.parse_args()
    for infile in get_filenames(args):
        if args.verbose:
            print('Processing: ', f)
        df = pd.read_csv(infile)
        with open(replace_ext(infile, args.new_ext), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['L', 'x0', 'k', 'b', 'lf', 'li', 'li2','Lge', 'N','der','L10','L90','r_squared'])
            for col in range(3,4):
                x, y = get_xy(df, col, 5000, x_col='location')

                # Find best guess using grid search
                Ls = np.linspace(0, 400, 10)
                x0s = np.linspace(0, 500, 10)
                ks = np.logspace(np.log10(0.001), np.log10(1), 10)
                bs = np.linspace(-10, 80, 10)
                grid = np.meshgrid(x, Ls, x0s, ks, bs, indexing='ij')
                yhat = sigmoid(*grid)

                # Mean-squared error
                mse = ((yhat - np.expand_dims(y, (1,2,3,4)))**2).mean(axis=0)
                idx = np.unravel_index(np.argmin(mse, axis=None), mse.shape)
                guess = [f[i] for f, i in zip((Ls, x0s, ks, bs), idx)]

                # Optimize guess using curve-fit
                cutoff = 25000
                cut = ~(np.isnan(x) | np.isnan(y) | x > cutoff)
                after_cut = ~(np.isnan(x) | np.isnan(y)| x <= cutoff)
                popt, pcov = curve_fit(sigmoid, x[cut], y[cut], guess, method='lm')
                r_squared = get_rsquared(sigmoid, x[cut], y[cut], popt)

                L, x0, k, b = popt

                sigmoid_tail = make_sigmoid_tail(*popt, cutoff)
                popt_tail, pcov_tail = curve_fit(sigmoid_tail, x[after_cut], y[after_cut], [0, 0], method='lm')

                plt.scatter(x, y)
                plt.plot(x, sigmoid(x, *popt))
                plt.plot(x[after_cut], sigmoid_tail(x[after_cut], *popt_tail))
                plt.savefig(replace_ext(infile, args.plot_ext))

                #Lge = -np.log(1/0.95-1)/k+x0
                Lge = -np.log(1/0.95-1)/k+x0
                L10 = -np.log(1/0.1-1)/k+x0
                L90 = -np.log(1/0.90-1)/k+x0
                #plt.axvline(x=Lge, c='r')
                lf = max(sigmoid(x, L, x0, k, b))
                li = (y[0]+y[1]+y[2])/3
                li2 = sigmoid(0, L, x0, k, b)
                N = integrate.quad(lambda x: 1/sigmoid(x, L, x0, k, b), 0, Lge)[0]
                #derivative
                der=k*L/4
                #(np.exp(-k*(x - x0))*k*L)/(1 + np.exp(-k*(x - x0)))**2
                writer.writerow([L, x0, k, b, lf, li, li2, Lge, N, der, L10, L90, r_squared])

if __name__=='__main__':
    main()
