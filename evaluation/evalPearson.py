import sys


from metrics import pearson_r


def read_preds(fh):
    return [float(l.strip()) for l in fh]


with open(sys.argv[1]) as gs_file:
    gs = read_preds(gs_file)

with open(sys.argv[2]) as sys_file:
    sys = read_preds(sys_file)

print(f'{pearson_r(sys, gs):.2f}')
