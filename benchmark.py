import torch
import sys

from time import time

origin = torch.rand(20, 120, 50)
size_list_below_100k = [1, 2, 3, 4, 5, 8, 10, 20, 50, 80, 100]
size_list_above_100k = [110, 120, 150, 180]

def discontigCopy(size_list):
    tensorx1000 = [origin[:, 2:2+item, :] for item in size_list]
    copy_time = {}
    inform  = "\ntest discontiguous copy\nsize\t\ttime(s)\n"
    for tensor_idx in range(len(size_list)):
        warmup = origin.clone()
        start_time = time()
        for iter_time in range(1000):
            y = tensorx1000[tensor_idx].clone()
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        copy_time[tensor_size] = interval 
    return inform, copy_time

def discontigAdd(size_list):
    tensorx1000 = [origin[:, 2:2+item, :] for item in size_list]
    add_time = {}
    inform  = "\ntest discontiguous add\nsize\t\ttime(s)\n"
    for tensor_idx in range(len(size_list)):
        warmup = origin.clone()
        start_time = time()
        for iter_time in range(1000):
            y = tensorx1000[tensor_idx] + 5.5 
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        add_time[tensor_size] = interval
    return inform, add_time

def contiguousCopy(size_list):
    tensorx1000 = [torch.rand(20, n, 50) for n in size_list]
    copy_time = {}
    inform  = "\ntest contiguous copy\nsize\t\ttime(s)\n"
    for tensor_idx in range(len(size_list)):
        warmup = tensorx1000[tensor_idx].clone()
        start_time = time()
        for iter_time in range(1000):
            y = warmup.clone()
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        copy_time[tensor_size] = interval
    return inform, copy_time

def contiguousAdd(size_list):
    tensorx1000 = [torch.rand(20, n, 50) for n in size_list]
    add_time = {}
    inform  = "\ntest contiguous add\nsize\t\ttime(s)\n"
    for tensor_idx in range(len(size_list)):
        warmup = tensorx1000[tensor_idx].clone()
        start_time = time()
        for iter_time in range(1000):
            y = warmup + 5.5 
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        add_time[tensor_size] = interval
    return inform, add_time

if __name__ == '__main__':
    try:
        test_benchmark = int(sys.argv[1])
        filename = sys.argv[2]
        with open(filename, 'a') as f:
            if(test_benchmark == 1):
                add_inform, add_time = contiguousAdd(size_list_above_100k)
                f.write(add_inform)
                for time in add_time:
                    f.write(time + '\t\t' + str(add_time[time]))
                    f.write('\n')
                f.write('\n')
            elif(test_benchmark == 2):
                copy_inform, copy_time = contiguousCopy(size_list_below_100k)
                add_inform, add_time = contiguousAdd(size_list_below_100k)
                f.write(copy_inform)
                for time in copy_time:
                    f.write(time + '\t\t' + str(copy_time[time]))
                    f.write('\n')
                f.write('\n')
                f.write(add_inform)
                for time in add_time:
                    f.write(time + '\t\t' + str(add_time[time]))
                    f.write('\n')
                f.write('\n')
            elif(test_benchmark == 3):
                copy_inform, copy_time = discontigCopy(size_list_below_100k)
                add_inform, add_time = discontigAdd(size_list_below_100k)
                f.write(copy_inform)
                for time in copy_time:
                    f.write(time + '\t\t' + str(copy_time[time]))
                    f.write('\n')
                f.write('\n')
                f.write(add_inform)
                for time in add_time:
                    f.write(time + '\t\t' + str(add_time[time]))
                    f.write('\n')
                f.write('\n')
    except IndexError:
        print('''
        error:
              three arguments are NECESSARY:
              arg1: an integer, 1:[benchmark 4.1] 2:[benchmark 4.2], 3:[benchmark 4.3]
              arg2: a string, output filename
              ''')
