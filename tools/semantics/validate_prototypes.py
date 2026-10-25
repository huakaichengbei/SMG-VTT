import argparse
from smgvtt.semantics.prototypes.storage.prototype_bank import load_description_json
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('path');args=parser.parse_args()
    m=load_description_json(args.path)
    print('categories:',len(m),'blank descriptions:',sum(not t.strip() for t in m.values()))
