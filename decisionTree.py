import copy
import numpy as np
import sys

class BinaryTreeNode:
    def __init__(self, key, depth, is_leaf):
        self.left = None
        self.right = None
        self.class_label = None
        self.split_attribute = None
        self.split_attribute_value = None
        self.value = key
        self.depth = depth
        self.is_leaf = is_leaf


def main():
    if len(sys.argv) < 7:
        print("Please give train_input file, test_input file, max_depth, train_out file, "
              "test_out file, and metrics_out file respectively in commandline arguments")
    train_input_file = sys.argv[1]
    test_input_file = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_output_file = sys.argv[4]
    test_output_file = sys.argv[5]
    metrics_output_file = sys.argv[6]
    with open(train_input_file, 'r') as f:
        train_data = f.readlines()
    attribute_map = create_attribute_index_to_name_map(train_data[0])
    decision_tree_root = train(data=train_data[1:], max_depth=max_depth, attribute_map=attribute_map)
    train_output_labels = predict(train_data[1:], decision_tree_root)
    with open(train_output_file, 'w') as f:
        for train_output_label in train_output_labels:
            f.write(train_output_label + "\n")
    with open(test_input_file, 'r') as f:
        test_data = f.readlines()
    test_output_labels = predict(test_data[1:], decision_tree_root)
    with open(test_output_file, 'w') as f:
        for test_output_label in test_output_labels:
            f.write(test_output_label + "\n")
    train_error = calculate_error_rate(train_output_labels, train_input_file)
    test_error = calculate_error_rate(test_output_labels, test_input_file)
    with open(metrics_output_file, 'w') as f:
        f.write("error(train): %s\n" % train_error)
        f.write("error(test): %s\n" % test_error)

def pretty_print_node(data, class_labels, depth, split_attribute=None, split_value=None):
    class_label_to_count_map = {class_labels[0]: 0, class_labels[1]: 0}
    for row in data:
        class_label_to_count_map[row.split()[-1]] += 1
    if depth == 0:
        print("[{} {}/{} {}]".format(class_label_to_count_map[class_labels[0]],
              class_labels[0], class_label_to_count_map[class_labels[1]], class_labels[1]))
    else:
        print(depth * "| " + "{} = {}: [{} {}/{} {}]".format(split_attribute, split_value,
                                                             class_label_to_count_map[class_labels[0]], class_labels[0],
                                                             class_label_to_count_map[class_labels[1]], class_labels[1]))


def create_attribute_index_to_name_map(info):
    attribute_names = [x for x in info.split()]
    index = 0
    attribute_map = {}
    for name in attribute_names:
        attribute_map[index] = name
        index += 1
    return attribute_map


def calculate_attribute_value_to_count_map(data, attribute_index):
    value_to_count_map = {}
    for row in data:
        attribute_value = row.split()[attribute_index]
        if attribute_value in value_to_count_map.keys():
            value_to_count_map[attribute_value] += 1
        else:
            value_to_count_map[attribute_value] = 1
    return value_to_count_map


def calculate_entropy(data, total_count):
    label_to_count_dict = calculate_attribute_value_to_count_map(data, -1)
    sum = 0
    for key in label_to_count_dict.keys():
        sum += np.multiply(float(label_to_count_dict[key])/float(total_count),
                           np.log2(float(label_to_count_dict[key])/float(total_count)))
    return np.multiply(-1, sum)


def calculate_specific_conditional_entropy(data, attribute_index, attribute_value):
    label_to_count_dict = {}
    attribute_value_count = 0
    for row in data:
        row_attribute_value = row.split()[attribute_index]
        row_label_value = row.split()[-1]
        if row_attribute_value == attribute_value:
            attribute_value_count += 1
            if row_label_value in label_to_count_dict.keys():
                label_to_count_dict[row_label_value] += 1
            else:
                label_to_count_dict[row_label_value] = 1
    sum = 0
    for key in label_to_count_dict.keys():
        sum += np.multiply(float(label_to_count_dict[key])/float(attribute_value_count),
                           np.log2(float(label_to_count_dict[key])/float(attribute_value_count)))
    return np.multiply(-1, sum)


def calculate_conditional_entropy(data, attribute_index, total_count):
    label_to_count_dict = calculate_attribute_value_to_count_map(data, attribute_index)
    sum = 0
    for key in label_to_count_dict.keys():
        sum += np.multiply(float(label_to_count_dict[key])/float(total_count),
                           calculate_specific_conditional_entropy(data, attribute_index, key))
    return sum


def find_splitting_attribute(data, suitable_splitting_attributes):
    if len(suitable_splitting_attributes) == 0:
        print("All splitting attributes are exhausted")
    total_count = len(data)
    entropy = calculate_entropy(data, total_count)
    mutual_info = calculate_mutual_info(data, entropy, suitable_splitting_attributes[0], total_count)
    attribute_index = suitable_splitting_attributes[0]
    for attribute in suitable_splitting_attributes[1:]:
        attribute_mutual_info = calculate_mutual_info(data, entropy, attribute, total_count)
        if attribute_mutual_info > mutual_info:
            mutual_info = attribute_mutual_info
            attribute_index = attribute
    return attribute_index


def calculate_mutual_info(data, entropy, attribute_index, total_count):
    conditional_entropy = calculate_conditional_entropy(data, attribute_index, total_count)
    mutual_info = entropy - conditional_entropy
    return mutual_info


def split_data_attribute_wise(data, split_attribute_index, split_attribute_values):
    split_data = {split_attribute_values[0]: [], split_attribute_values[1]: []}
    for row in data:
        if row.split()[split_attribute_index] == split_attribute_values[0]:
            split_data[split_attribute_values[0]].append(row)
        else:
            split_data[split_attribute_values[1]].append(row)
    return [split_data[split_attribute_values[0]], split_data[split_attribute_values[1]]]


def train(data, max_depth, attribute_map):
    if len(data) == 0:
        print("No input data to train.")
    suitable_splitting_attributes = list(attribute_map.keys())[:-1]
    value = {'data': data, 'suitable_splitting_attributes': suitable_splitting_attributes}
    root_node = BinaryTreeNode(key=value, depth=0, is_leaf=False)
    class_labels = calculate_attribute_value_to_count_map(data, -1).keys()
    pretty_print_node(data, class_labels, 0)
    train_tree(root_node, max_depth, class_labels, attribute_map)
    return root_node


def train_tree(node, max_depth, class_labels, attribute_map):
    if (node.depth >= max_depth) or (node.value['suitable_splitting_attributes'] == []):
        node.is_leaf = True
        label_to_count_map = calculate_attribute_value_to_count_map(node.value['data'], -1)
        if len(label_to_count_map.keys()) == 1:
            node.class_label = list(label_to_count_map.keys())[0]
        else:
            node.class_label = max(label_to_count_map, key=label_to_count_map.get)

    else:
        split_attribute_index = find_splitting_attribute(node.value['data'],
                                                         node.value['suitable_splitting_attributes'])
        node.split_attribute = split_attribute_index
        suitable_splitting_attributes = copy.deepcopy(node.value['suitable_splitting_attributes'])
        suitable_splitting_attributes.remove(split_attribute_index)
        split_attribute_values = calculate_attribute_value_to_count_map(node.value['data'],
                                                                        split_attribute_index).keys()
        if len(split_attribute_values) == 1:
            label_to_count_map = calculate_attribute_value_to_count_map(node.value['data'], -1)
            node.class_label = list(label_to_count_map.keys())[0]
            return
        split_data = split_data_attribute_wise(node.value['data'], split_attribute_index, split_attribute_values)
        left_val = {'data': split_data[0],
                    'suitable_splitting_attributes': suitable_splitting_attributes}
        right_val = {'data': split_data[1],
                     'suitable_splitting_attributes': suitable_splitting_attributes}
        if len(split_data[0]) != 0:
            node.left = BinaryTreeNode(key=left_val, depth=node.depth + 1, is_leaf=False)
            node.left.split_attribute_value = split_attribute_values[0]
            pretty_print_node(split_data[0], class_labels, node.depth + 1,
                              split_attribute=attribute_map[split_attribute_index],
                              split_value=split_attribute_values[0])
            train_tree(node.left, max_depth, class_labels, attribute_map)
        if len(split_data[1]) != 0:
            node.right = BinaryTreeNode(key=right_val, depth=node.depth + 1, is_leaf=False)
            node.right.split_attribute_value = split_attribute_values[1]
            pretty_print_node(split_data[1], class_labels, node.depth + 1,
                              split_attribute=attribute_map[split_attribute_index],
                              split_value=split_attribute_values[1])
            train_tree(node.right, max_depth, class_labels, attribute_map)


def predict(data, decision_tree):
    predicted_labels = []
    for row in data:
        row = row.split()
        predicted_labels.append(predict_row(row, decision_tree))
    return predicted_labels


def predict_row(data, decision_tree_node):
    if decision_tree_node.is_leaf:
        label = decision_tree_node.class_label
        return label
    else:
        split_attribute_index = decision_tree_node.split_attribute
        split_attribute_value = data[split_attribute_index]
        if decision_tree_node.left and decision_tree_node.left.split_attribute_value == split_attribute_value:
            return predict_row(data, decision_tree_node.left)
        elif decision_tree_node.right and decision_tree_node.right.split_attribute_value == split_attribute_value:
            return predict_row(data, decision_tree_node.right)
        else:
            return decision_tree_node.class_label


def calculate_error_rate(predicted_labels, actual_labels_file):
    count = 0
    print(predicted_labels)
    with open(actual_labels_file, 'r') as f:
        data = f.readlines()
    actual_labels = [row.split()[-1] for row in data[1:]]
    print(actual_labels)
    for i in range(0, len(actual_labels)):
        if actual_labels[i] != predicted_labels[i]:
            count += 1
    print(count)
    print(len(actual_labels))
    return float(count)/float(len(actual_labels))


if __name__ == "__main__":
    main()
