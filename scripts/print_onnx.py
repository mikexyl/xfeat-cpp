import onnx
import sys

def print_onnx_structure(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    print("Model Inputs:")
    for input_tensor in graph.input:
        print(f"  {input_tensor.name}: {input_tensor.type}")

    print("\nModel Outputs:")
    for output_tensor in graph.output:
        print(f"  {output_tensor.name}: {output_tensor.type}")

    print("\nModel Nodes:")
    for node in graph.node:
        print(f"  {node.op_type}: {node.name}")
        print(f"    Inputs: {node.input}")
        print(f"    Outputs: {node.output}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_onnx.py <model.onnx>")
    else:
        print_onnx_structure(sys.argv[1])
