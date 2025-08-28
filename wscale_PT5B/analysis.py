import pandas
import ast
from scipy.interpolate import interp1d
import pickle


def parse_sec_loc(sec_loc):
    return ast.literal_eval(sec_loc)


EPSPNORM = 0.5
df = pandas.read_csv('grid_search.csv')[['sec', 'weight', 'epsp']]
sec_locs = [[i, 0.5] for i in df['sec']]
df[['sec', 'loc']] = pandas.DataFrame(sec_locs, index=df.index)
secs = df['sec'].unique()

wnorms = {}

for sec in secs:
    locs = df[df['sec'] == sec]['loc'].unique()

    # for each section calculate the weight where the epsp at soma == 0.5
    entries = df[df['sec'] == sec].sort_values(by='weight')
    # print(entries)
    weights = entries['weight']
    epsps = entries['epsp']
    f = interp1d(epsps, weights, fill_value='extrapolate')
    # print([*zip(weights, epsps)])
    w = f(EPSPNORM) 
    while w < 0:
        x_new, y_new = zip(*epspSeg[:-1])
        f = interp1d(y_new, x_new, fill_value="extrapolate")
        w = f(EPSPNORM)
    wnorm = f(EPSPNORM) / EPSPNORM
    wnorms[sec] = [wnorm]

filename = 'PT5B_full_weightNorm_TIM.pkl' # 'weight_norms.pkl'
print(wnorms)
with open(filename, 'wb') as fptr:
    pickle.dump(wnorms, fptr)