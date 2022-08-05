import onnx

from up.utils.general.log_helper import default_logger as logger


def dfs_search_reachable_nodes(G, node_output_name, graph_input_names, reachable_nodes):
    if node_output_name in graph_input_names:
        return
    for node in G.graph.node:
        if node in reachable_nodes:
            continue
        if node_output_name not in node.output:
            continue
        reachable_nodes.append(node)
        for name in node.input:
            dfs_search_reachable_nodes(G, name, graph_input_names, reachable_nodes)


def collect_subnet_nodes(G, input_names, output_names):
    reachable_nodes = []
    for name in output_names:
        dfs_search_reachable_nodes(G, name, input_names, reachable_nodes)
    # needs to be topology sorted.
    nodes = [n for n in G.graph.node if n in reachable_nodes]
    return nodes


def collect_subnet_inits_values(nodes, wmap, vimap):
    all_tensors_name = set()
    for node in nodes:
        for name in node.input:
            all_tensors_name.add(name)
        for name in node.output:
            all_tensors_name.add(name)
    initializer = [wmap[t] for t in wmap.keys() if t in all_tensors_name]
    value_info = [vimap[t] for t in vimap.keys() if t in all_tensors_name]
    return initializer, value_info


def build_name_dict(objs):
    return {obj.name: obj for obj in objs}


def auto_remove_cast(G):
    logger.info("remove useless cast layers")
    cast_names_list = []
    name_node_dict = {}

    for node in G.graph.node:
        name_node_dict[node.name] = node
        if 'Cast' in node.name:
            cast_names_list.append(node.name)

    graph_input_names = []
    for inp in G.graph.input:
        graph_input_names.append(inp.name)
    graph_output_names = []
    for oup in G.graph.output:
        graph_output_names.append(oup.name)

    for cast_name in cast_names_list:
        logger.info('process {}'.format(cast_name))
        cast_node = name_node_dict[cast_name]
        cast_input = cast_node.input[0]
        cast_output = cast_node.output[0]

        if cast_input in graph_input_names:
            # search next node
            for idx, node in enumerate(G.graph.node):
                if cast_output in node.input and 'Cast' not in node.name:
                    next_node = node
                    logger.info('find {}'.format(node.name))
                    break
            G.graph.node.remove(cast_node)
            G.graph.node.remove(next_node)
            new_node_input = []
            for inp in next_node.input:
                if inp != cast_node.output[0]:
                    new_node_input.append(inp)
                else:
                    new_node_input.append(cast_node.input[0])

            new_node = onnx.helper.make_node(
                name=next_node.name,
                op_type=next_node.op_type,
                inputs=new_node_input,
                outputs=next_node.output,
            )
            for att_id, attr in enumerate(next_node.attribute):
                new_node.attribute.append(attr)
            G.graph.node.insert(idx, new_node)
        else:
            # not the after input cast
            for idx, node in enumerate(G.graph.node):
                if cast_input in node.output:
                    prev_node = node
                    logger.info('find {}'.format(node.name))
                    break
            G.graph.node.remove(cast_node)
            G.graph.node.remove(prev_node)
            new_node = onnx.helper.make_node(
                name=prev_node.name,
                op_type=prev_node.op_type,
                inputs=prev_node.input,
                outputs=cast_node.output,
            )
            for att_id, attr in enumerate(prev_node.attribute):
                new_node.attribute.append(attr)
            G.graph.node.insert(idx, new_node)
    return G


def make_constant_dims(name, shapes):
    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=[name],
        value=onnx.helper.make_tensor("val", onnx.TensorProto.INT64, [len(shapes), ], shapes),
    )
    return node
