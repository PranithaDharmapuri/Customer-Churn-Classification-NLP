def find_missing(input_list, reference_list):
    return list(set(reference_list) - set(input_list))


# Example
input_list = [1, 2, 3, 5]
reference_list = [1, 2, 3, 4, 5, 6]

missing = find_missing(input_list, reference_list)
print("Missing elements:", missing)
