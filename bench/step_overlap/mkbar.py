import json,re,collections,sys
S=sys.argv[1]
m=json.load(open(S+'/workloads/train.json'))
g=m['graphs'][0]; T=g['tasks']
step=lambda t:int(re.match(r's(\d+)/',t['id']).group(1))
by=collections.defaultdict(list)
for t in T: by[step(t)].append(t)
out=[]
for k in sorted(by):
    if k>0:
        b={'id':f's{k}/barrier','op':'noop','args':{},'after':[t['id'] for t in by[k-1]]}
        out.append(b)
        for t in by[k]: t['after']=list(t.get('after',[]))+[b['id']]
    out.extend(by[k])
g['tasks']=out
json.dump(m,open(S+'/workloads/train_bar.json','w'))
