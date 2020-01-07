import argparse
import re
import sys

parser = argparse.ArgumentParser()

parser.add_argument('wa_input')
parser.add_argument('scores_input')

args = parser.parse_args()

NOT_ALIGNED = '-not aligned-'

with open(args.wa_input) as wfp, open(args.scores_input) as sfp:
    scores = list(reversed([int(round(float(l.strip()))) for l in sfp]))

    for line in wfp:
        m = re.search(r'(//\s*)[0-9](\s*//\s*.*\s*<==>\s*.*\s*$)', line)
        if m and NOT_ALIGNED not in m.group(2):
            try:
                line = re.sub(r'(//\s*)[0-9](\s*//\s*.*\s*<==>\s*.*\s*$)', rf'\g<1>{scores.pop()}\2', line)
            except IndexError:
                print('WARNING: to few scores', file=sys.stderr)
                exit(1)
        print(line)

    if scores:
        print('WARNING: to much scores', file=sys.stderr)
        exit(1)
