import json, sys
r = json.load(open(sys.argv[1]))[0]
sys.exit(0 if any(x > 1.10 * min(t) for t in (r['ms']['control']['all'], r['ms']['cand']['all']) for x in t) else 1)
