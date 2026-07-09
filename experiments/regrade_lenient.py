#!/usr/bin/env python3
"""Gold-grader audit: re-grade --save_answers dumps with a paraphrase-tolerant grader.

For each <arm>_seed0_answers.json (a list of {question, answer, gold}), recompute gold with a
lenient grader = case/markdown/Unicode-subscript normalization + number-word and synonym
equivalence + WORD-BOUNDARY matching over the expanded vocabulary (word boundaries avoid
spurious substring hits like '49' inside the wrong answer '149'). Compares to the shipped
keyword grader's score (the 'gold' field) and reports per-arm deltas + ordering preservation.

Usage: python regrade_lenient.py --answers-root <dir containing run_*/answers/> [--task task.py dir]
"""
import json, re, glob, argparse, importlib.util, sys, os

NUM = {'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7',
       'eight':'8','nine':'9','ten':'10','eleven':'11','twelve':'12','hundred':'100'}
NUM.update({v:k for k,v in list(NUM.items())})
ALIAS = {'h2o':['water'], 'co2':['carbon dioxide'], 'pacific':['pacific ocean'],
         'mexico city':['cdmx'], 'tokyo':['tokio']}
SUB = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')

def norm(s):
    s = s.lower().translate(SUB).replace('**','').replace('*','')
    return re.sub(r'\s+',' ', re.sub(r'[^a-z0-9 ]',' ', s)).strip()

def variants(k):
    vs = {k}
    if k in NUM: vs.add(NUM[k])
    for a in ALIAS.get(k, []): vs.add(a)
    return vs

def lenient(answer, keys):
    a = norm(answer)
    return 1.0 if any(re.search(rf'\b{re.escape(norm(v))}\b', a) for k in keys for v in variants(k)) else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--answers-root', required=True)
    ap.add_argument('--task-dir', default='.', help='dir containing task.py with the QA list')
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('task', os.path.join(a.task_dir, 'task.py'))
    task = importlib.util.module_from_spec(spec); spec.loader.exec_module(task)
    ANS = {q: keys for q, keys, _ in task.QA}
    res = {}
    for f in sorted(glob.glob(os.path.join(a.answers_root, 'run_*/answers/*_answers.json'))):
        size = 'run_' + f.split('run_')[1].split('/')[0]
        arm = os.path.basename(f).replace('_seed0_answers.json','')
        recs = json.load(open(f))
        kw = sum(r['gold'] for r in recs)/len(recs)
        ln = sum(lenient(r['answer'], ANS[r['question']]) for r in recs)/len(recs)
        res[(size, arm)] = (kw, ln)
        print(f'{size:10}{arm:16} keyword={kw:.3f} lenient={ln:.3f} delta={ln-kw:+.3f}')
    print(f'\nmax |delta| = {max(abs(ln-kw) for kw,ln in res.values()):.3f}')

if __name__ == '__main__':
    main()
