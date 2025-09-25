#!/usr/bin/env python3
import os, json

checks = []
files = [
 ('src/evaluation/metrics.py','Metrics utility'),
 ('src/evaluation/visualizer.py','Visualization utility'),
 ('src/evaluation/evaluator.py','Evaluator utility'),
 ('scripts/phase4_evaluation.py','Evaluation script')
]
for fp,desc in files:
    checks.append((os.path.exists(fp),desc))

print('🔍 Phase 4 Status')
print('='*20)
for ok,desc in checks:
    print(('✅' if ok else '❌'),desc)
