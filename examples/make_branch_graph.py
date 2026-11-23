"""Generate a tiny ONNX graph with repeated operand uses and a branch join."""
from pathlib import Path
import argparse


def main():
    import onnx
    from onnx import helper, TensorProto
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    info = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 3])
    nodes = [helper.make_node('Mul', ['x','x'], ['a']),
             helper.make_node('Add', ['a','x'], ['b']),
             helper.make_node('Add', ['a','b'], ['y'])]
    graph = helper.make_graph(nodes, 'branch_join', [info('x')], [info('y')],
                              value_info=[info('a'),info('b')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',18)])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,args.output)


if __name__ == '__main__':
    main()
