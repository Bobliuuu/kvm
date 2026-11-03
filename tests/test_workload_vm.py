import sys, time, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from workload_vm import Graph, GraphError, Registry, Runtime, Store, default_registry

class VMTests(unittest.TestCase):
    def run_graphs(self, graphs, state=None, workers=4, registry=None):
        vm = Runtime(registry or default_registry(), Store(state or {}), workers)
        vm.start()
        try:
            for g in graphs: vm.submit(Graph.from_dict(g))
            trace = vm.wait(timeout=3)
            return vm.store.data, trace
        finally:
            vm.shutdown()

    def test_training_graph(self):
        g = {'name':'train','tasks':[
            {'id':'forward','op':'predict','args':{'input':'x','weight':'w','out':'p'}},
            {'id':'backward','op':'mse_grad','after':['forward'],
             'args':{'pred':'p','target':'y','input':'x','loss':'loss','grad':'grad'}},
            {'id':'update','op':'sgd','after':['backward'],
             'args':{'weight':'w','grad':'grad','lr':0.1}}]}
        state, trace = self.run_graphs([g],{'x':2,'w':0.5,'y':6})
        self.assertAlmostEqual(state['w'],2.5)
        self.assertEqual([t['task'] for t in trace],['train/forward','train/backward','train/update'])

    def test_parallel_independent(self):
        g = {'name':'parallel','tasks':[
            {'id':str(i),'op':'scale','args':{'src':'x','factor':i,'out':f'out{i}'}}
            for i in range(100)]}
        state,trace = self.run_graphs([g],{'x':3})
        self.assertEqual(len(trace),100)
        self.assertEqual([state[f'out{i}'] for i in range(100)], [3*i for i in range(100)])

    def test_submit_while_running(self):
        r=default_registry()
        r.register('sleep', lambda s,a: time.sleep(a['duration']))
        vm=Runtime(r,Store({'x':2}),workers=2)
        vm.start()
        try:
            vm.submit(Graph.from_dict({'name':'background','tasks':[
                {'id':'slow','op':'sleep','args':{'duration':0.02}}]}))
            vm.submit(Graph.from_dict({'name':'request','tasks':[
                {'id':'fast','op':'scale','args':{'src':'x','factor':5,'out':'result'},'priority':10}]}))
            vm.wait(timeout=2)
            self.assertEqual(vm.store.get('result'),10)
        finally: vm.shutdown()

    def test_extensible_operator(self):
        r=Registry()
        r.register('custom', lambda s,a: s.put('result',a['value']**3))
        state,_=self.run_graphs([{'name':'custom_graph','tasks':[
            {'id':'run','op':'custom','args':{'value':4}}]}],registry=r)
        self.assertEqual(state['result'],64)

    def test_unknown_operator(self):
        vm=Runtime(default_registry(),Store())
        with self.assertRaises(GraphError):
            vm.submit(Graph.from_dict({'name':'x','tasks':[{'id':'a','op':'doom'}]}))

    def test_validation(self):
        cases=[
            [{'id':'a','op':'copy','after':['b']},{'id':'b','op':'copy','after':['a']}],
            [{'id':'a','op':'copy','after':['missing']}],
            [{'id':'a','op':'copy'},{'id':'a','op':'copy'}],
            [{'id':'a','op':'copy','after':['b','b']},{'id':'b','op':'copy'}],
        ]
        for tasks in cases:
            with self.subTest(tasks=tasks), self.assertRaises(GraphError):
                Graph.from_dict({'name':'bad','tasks':tasks})

    def test_failure_diamond_cancels_once(self):
        r=Registry()
        r.register('fail',lambda s,a: 1/0)
        r.register('noop',lambda s,a: None)
        vm=Runtime(r,Store(),workers=2)
        vm.start()
        try:
            vm.submit(Graph.from_dict({'name':'diamond','tasks':[
                {'id':'a','op':'fail'},
                {'id':'b','op':'noop','after':['a']},
                {'id':'c','op':'noop','after':['a']},
                {'id':'d','op':'noop','after':['b','c']}]}))
            with self.assertRaises(RuntimeError): vm.wait(timeout=2)
            self.assertEqual(len(vm.completed),4)
            self.assertEqual(len(set(vm.completed)),4)
        finally: vm.shutdown()

    def test_failure_cancels_dependents(self):
        r=Registry()
        r.register('fail',lambda s,a: 1/0)
        r.register('noop',lambda s,a: None)
        vm=Runtime(r,Store(),workers=2)
        vm.start()
        try:
            vm.submit(Graph.from_dict({'name':'failgraph','tasks':[
                {'id':'a','op':'fail'}, {'id':'b','op':'noop','after':['a']},
                {'id':'c','op':'noop','after':['b']}]}))
            with self.assertRaises(RuntimeError): vm.wait(timeout=2)
            self.assertEqual(len(vm.completed),3)
        finally: vm.shutdown()

if __name__=='__main__': unittest.main()
