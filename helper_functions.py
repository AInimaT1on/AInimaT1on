import os
import cv2
import datetime as dt

def calc_accuracy(mdl, test_data):
    # reduce/collapse the classification dimension according to max op
    # resulting in most likely label
    total_acc = []
    for images, labels in iter(test_data):
        #images.resize_(images.size()[0],784)
        max_vals, max_indices = mdl(images).max(1)
        # assumes the first dimension is batch size
        n = max_indices.size(0)  # index 0 for extracting the # of elements
        # calulate acc (note .item() to do float division)
        acc = (max_indices == labels).sum().item() / n
        total_acc.append(acc)

    final_acc = sum(total_acc) / len(total_acc)
    print(f"The average accuracy across all tests: {final_acc}, test_size: {len(total_acc)}")
    return final_acc




def save_sign_img(sign_label, img):
    now_code = dt.datetime.now().strftime("%y%m%d%H%M%S")
    new_file_name =  sign_label + "/" + sign_code + "_" + now_code + ".jpg"
    duplicate_counter = 1
    while os.path.exists(new_file_name):
        new_file_name = sign_label + "/" + sign_code + now_code + f"_{str(duplicate_counter)}" + ".jpg"

def save_sign_img(sign_label, img):
    if not os.path.exists(f'data/data_collector/rcd_tmp/{sign_label}'):
        os.makedirs(f'data/data_collector/rcd_tmp/{sign_label}')
    now_code = dt.datetime.now().strftime("%y%m%d%H%M%S")
    new_file_name =  f"data/data_collector/rcd_tmp/{sign_label}/{sign_label}_{now_code}.jpg"
    duplicate_counter = 1
    while os.path.exists(new_file_name):
        new_file_name = f"data/data_collector/rcd_tmp/{sign_label}/{sign_label}_{now_code}_{duplicate_counter}.jpg"

        duplicate_counter += 1
    cv2.imwrite(new_file_name, img)
