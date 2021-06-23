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
        duplicate_counter += 1
    cv2.imwrite(new_file_name, img)

def number_to_letter(prediction):
    alphabet_dic = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',
    11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X'
    ,24:'Y',25:'Z'}
    try:        
        res = alphabet_dic[prediction.item()]
        return (res)
    except:
        print('Error in number_to_letter function')
        return str(prediction)