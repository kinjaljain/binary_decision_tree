import math
import sys

def main():
    if len(sys.argv) < 3:
        print("Please give both input and output file names")
    input_file, output_file = sys.argv[1], sys.argv[2]
    with open(input_file, 'r') as f:
        train_data = f.readlines()
    count_dict = create_count_dict(train_data[1:])
    print("Count_dict: %s" % count_dict)
    entropy = calculate_entropy(count_dict)
    print("Entropy: %s" % entropy)
    majority_class = find_majority_class(count_dict)
    print("Majority class: %s" % majority_class)
    error_rate = calculate_error_rate(train_data[1:], majority_class)
    print("Error rate: %s" % error_rate)
    with open(output_file, 'w') as f:
        f.write("entropy: %s\n" % entropy)
        f.write("error: %s\n" % error_rate)

def create_count_dict(lines):
    count_dict = {}
    for line in lines:
        party = line.split()[-1]
        if party in count_dict.keys():
            count_dict[party] += 1
        else:
            count_dict[party] = 1
    return count_dict

def calculate_entropy(count_dict):
    total_count = 0
    entropy = 0
    for key in count_dict.keys():
        total_count += count_dict[key]
    for key in count_dict.keys():
        entropy += -((float(count_dict[key])/float(total_count)) *
                     math.log(float(count_dict[key])/float(total_count), 2))
    return entropy

def find_majority_class(count_dict):
    print(count_dict.keys())
    majority_class = list(count_dict.keys())[0]
    for key in count_dict.keys():
        if count_dict[key] > count_dict[majority_class]:
            majority_class = key
    return majority_class

def calculate_error_rate(lines, majority_class):
    total_count = len(lines)
    error_count = 0
    for line in lines:
        if line.split()[-1] != majority_class:
            error_count += 1
    return float(error_count)/float(total_count)

if __name__ == "__main__":
    main()
