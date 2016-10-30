
from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput


config = Config(max_depth=3)
graphviz = GraphvizOutput(output_file='filter_max_depth.dot')

with PyCallGraph(output=graphviz, config=config):
    from egovision import handDetection
