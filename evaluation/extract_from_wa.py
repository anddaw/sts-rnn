import argparse
import re
from typing import Optional, Tuple


def parse_line(l: str) -> Tuple[Optional[str], Optional[str]]:

    m = re.search(r'//\s*(?P<score>[0-9])\s*//\s*(?P<s1>.*)\s*<==>\s*(?P<s2>.*)\s*$', l)

    if m:
        groups = m.groupdict()
        return '\t'.join([groups['s1'], groups['s2']]), groups['score']
    else:
        return None, None


parser = argparse.ArgumentParser()

parser.add_argument('input')
parser.add_argument('utterances_output')
parser.add_argument('scores_output')

args = parser.parse_args()

with open(args.input) as ifp, open(args.utterances_output, 'w') as ufp, open(args.scores_output, 'w') as sfp:
    for line in ifp:
        utterances, score = parse_line(line)

        if utterances and score:
            ufp.write(f'{utterances}\n')
            sfp.write(f'{score}\n')
