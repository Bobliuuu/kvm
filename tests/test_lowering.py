import sys, unittest, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from lower import Task, lower, validate_schedule, graph_to_csr, lower_json

class LoweringTests(unittest.TestCase):
    def test_diamond(self):
        tasks = [Task(signals=(0,0)), Task(waits=((0,1),),signals=(1,)),
                 Task(waits=((0,2),),signals=(1,)), Task(waits=((1,2),))]
        g = lower(tasks)
        self.assertEqual(g.indegree,[0,1,1,2])
        validate_schedule(g,[0,1,2,3])
        with self.assertRaises(ValueError): validate_schedule(g,[0,3,1,2])

    def test_deduplicate_wait_edges(self):
        g = lower([Task(signals=(0,1)),Task(waits=((0,1),(1,1)))])
        self.assertEqual(g.edges,1)

    def test_zero_threshold(self):
        g = lower([Task(waits=((4,0),))])
        self.assertEqual(g.indegree,[0])

    def test_missing_producer(self):
        with self.assertRaisesRegex(ValueError,'only 0 available'):
            lower([Task(waits=((9,1),))])

    def test_self_dependency(self):
        with self.assertRaisesRegex(ValueError,'self-dependency'):
            lower([Task(waits=((0,1),),signals=(0,))])

    def test_cycle(self):
        with self.assertRaisesRegex(ValueError,'cyclic'):
            lower([Task(waits=((1,1),),signals=(0,)),
                   Task(waits=((0,1),),signals=(1,))])

    def test_negative(self):
        with self.assertRaises(ValueError): lower([Task(signals=(-1,))])
        with self.assertRaises(ValueError): lower([Task(waits=((1,-1),))])

    def test_csr(self):
        g=lower([Task(signals=(0,)),Task(waits=((0,1),)),Task(waits=((0,1),))])
        self.assertEqual(graph_to_csr(g),([0,2,2,2],[1,2]))

    def test_json_roundtrip(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/"tape.json"; dst=Path(d)/"dag.json"
            src.write_text(json.dumps({"tasks":[{"signals":[3]},{"waits":[[3,1]]}]}))
            lower_json(str(src),str(dst))
            dag=json.loads(dst.read_text())
            self.assertEqual(dag["indegree"],[0,1])
            self.assertEqual(dag["initial_ready"],[0])

    def test_random_chain_and_fanout(self):
        rng=random.Random(7)
        for _ in range(200):
            n=rng.randrange(2,70)
            tasks=[Task(signals=(i,), waits=((i-1,1),) if i else ()) for i in range(n)]
            g=lower(tasks)
            self.assertEqual(g.edges,n-1)
            validate_schedule(g,list(range(n)))
            with self.assertRaises(ValueError):
                validate_schedule(g,list(reversed(range(n))))

if __name__ == '__main__': unittest.main()
