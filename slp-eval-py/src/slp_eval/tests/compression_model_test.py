from slp_eval.compression_model import SLP


def test_super_compress():
    for n in range(10):
        slp = SLP(
            constants=["a"],
            instructions=[(i,i) for i in range(n)],
        ) 
        v = slp.evaluate()
        assert (all(x == "a" for x in v)) 
        assert len(v) == 2**n


