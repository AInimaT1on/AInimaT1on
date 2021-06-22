def calc_accuracy(mdl, test_data):
    # reduce/collapse the classification dimension according to max op
    # resulting in most likely label
    total_acc = []
    for images, labels in iter(test_data):
        images.resize_(images.size()[0],784)
        max_vals, max_indices = mdl(images).max(1)
        # assumes the first dimension is batch size
        n = max_indices.size(0)  # index 0 for extracting the # of elements
        # calulate acc (note .item() to do float division)
        acc = (max_indices == labels).sum().item() / n
        total_acc.append(acc)

    final_acc = sum(total_acc) / len(total_acc)
    print(f"The average accuracy across all tests: {final_acc}, test_size: {len(total_acc)}")
    return final_acc
