import sys, json, struct, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from compile_workload import compile_spec, write_compiled, CompileError, DESC

class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'noop': 0, 'custom': 9}
        self.spec = {'graphs': [{'name':'a','tasks':[
            {'id':'start','op':'noop'},
            {'id':'left','op':'custom','after':['start'],'priority':10},
            {'id':'right','op':'custom','after':['start']},
            {'id':'end','op':'noop','after':['left','right']}
        ]}]}

    def test_diamond_abi(self):
        m, desc, succ, payload = compile_spec(self.spec,self.manifest)
        self.assertEqual((m['task_count'],m['edge_count'],len(desc)),(4,4,128))
        d=[DESC.unpack_from(desc,i*DESC.size) for i in range(4)]
        self.assertEqual([x[3] for x in d],[0,1,1,2])
        self.assertEqual([x[2] for x in d],[2,1,1,0])
        self.assertEqual(d[1][4],10)
        self.assertEqual(struct.unpack('<4i',succ),(1,2,3,3))
        self.assertEqual(json.loads(payload[d[0][6]:d[0][6]+d[0][7]]),{})

    def test_determinism(self):
        self.assertEqual(compile_spec(self.spec,self.manifest),compile_spec(self.spec,self.manifest))

    def test_binary_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            m=write_compiled(self.spec,self.manifest,d)
            self.assertEqual(json.loads((Path(d)/'metadata.json').read_text()),m)
            self.assertEqual((Path(d)/'tasks.bin').stat().st_size,4*32)

    def test_unknown_op(self):
        self.spec['graphs'][0]['tasks'][0]['op']='missing'
        with self.assertRaises(CompileError): compile_spec(self.spec,self.manifest)

    def test_missing_dependency(self):
        self.spec['graphs'][0]['tasks'][1]['after']=['missing']
        with self.assertRaises(CompileError): compile_spec(self.spec,self.manifest)

    def test_cycle(self):
        self.spec['graphs'][0]['tasks'][0]['after']=['end']
        with self.assertRaises(CompileError): compile_spec(self.spec,self.manifest)

    def test_duplicate(self):
        self.spec['graphs'][0]['tasks'].append(self.spec['graphs'][0]['tasks'][0])
        with self.assertRaises(CompileError): compile_spec(self.spec,self.manifest)

    def test_bad_manifest(self):
        with self.assertRaises(CompileError): compile_spec(self.spec,{'a':1,'b':1})

    def test_invalid_deadline(self):
        self.spec['graphs'][0]['tasks'][0]['deadline_us']=1.5
        with self.assertRaises(CompileError): compile_spec(self.spec,self.manifest)

    def test_custom_operator_is_generic(self):
        self.spec['graphs'][0]['tasks'][1]['args']={'shader':'unrecognized','tiles':[1,2]}
        m,d,s,p=compile_spec(self.spec,self.manifest)
        descriptor=DESC.unpack_from(d,32)
        self.assertEqual(json.loads(p[descriptor[6]:descriptor[6]+descriptor[7]]),
                         {'shader':'unrecognized','tiles':[1,2]})

if __name__=='__main__': unittest.main()
