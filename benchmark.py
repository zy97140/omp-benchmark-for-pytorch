import torch
import sys
from time import time

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("continuity", metavar="continuity",
                    help="display operands' continuity, can only be chosen from contiguous and discontiguous",
                    choices=["contiguous", "discontiguous"])
parser.add_argument("operation", metavar="operation",
                    help="display an elementwise operation, can only be chosen from copy, add, div, exp, sin, sum and prod",
                    choices=["copy", "add", "div", "exp", "sin", "sum", "prod"])
parser.add_argument("--o", metavar="output filename", help="display ouput filename", default="output.log", dest="outputfile")
args = parser.parse_args()


origin = torch.rand(20, 200, 50)
size_list = [1, 2, 3, 4, 5, 8, 10, 20, 50, 80, 100, 110, 120, 150, 180]

def discontigCopy(size_list):
    tensorx1000 = [origin[:, 2:2+item, :] for item in size_list]
    copy_time = {}
    inform  = "\ntest discontiguous copy\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = origin.clone()
        for iter_time in range(1000):
            y = tensorx1000[tensor_idx].clone()
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
    inform  = "\ntest discontiguous add\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = origin.clone()
        for iter_time in range(1000):
            y = tensorx1000[tensor_idx] + 5.5 
        start_time = time()
        for iter_time in range(1000):
            y = tensorx1000[tensor_idx] + 5.5 
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        add_time[tensor_size] = interval
    return inform, add_time

def discontigDiv(size_list):
    tensorx1000 = [origin[:, 2:2+item, :] for item in size_list]
    div_time = {}
    inform = "\ntest discontiguous div\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = origin.clone()
        for iter_time in range(1000):
            y = torch.div(tensorx1000[tensor_idx], 3)
        start_time = time()
        for iter_time in range(1000):
            y = torch.div(tensorx1000[tensor_idx], 3)
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = "%dk" % size_list[tensor_idx]
        div_time[tensor_size] = interval
    return inform, div_time

def discontigSin(size_list):
    tensorx1000 = [origin[:, 2:2+item, :] for item in size_list]
    sin_time = {}
    inform = "\ntest discontiguous sin\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = origin.clone()
        for iter_time in range(1000):
            y = torch.sin(tensorx1000[tensor_idx])
        start_time = time()
        for iter_time in range(1000):
            y = torch.sin(tensorx1000[tensor_idx])
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        sin_time[tensor_size] = interval
    return inform, sin_time

def discontigExp(size_list):
    tensorx1000 = [origin[:, 2:2+item, :] for item in size_list]
    exp_time = {}
    inform = "\ntest discontiguous exp\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = origin.clone()
        for iter_time in range(1000):
            y = torch.exp(tensorx1000[tensor_idx])
        start_time = time()
        for iter_time in range(1000):
            y = torch.exp(tensorx1000[tensor_idx])
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        exp_time[tensor_size] = interval
    return inform, exp_time

def discontigSum(size_list):
    tensorx1000 = [origin[:, 2:2+item, :] for item in size_list]
    sum_time = {}
    inform = "\ntest discontiguous sum\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = origin.clone()
        for iter_time in range(1000):
            y = torch.sum(tensorx1000[tensor_idx], 0)
        start_time = time()
        for iter_time in range(1000):
            y = torch.sum(tensorx1000[tensor_idx], 0)
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        sum_time[tensor_size] = interval
    return inform, sum_time

def discontigProd(size_list):
    tensorx1000 = [origin[:, 2:2+item, :] for item in size_list]
    prod_time = {}
    inform = "\ntest discontiguous prod\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = origin.clone()
        for iter_time in range(1000):
            y = torch.prod(tensorx1000[tensor_idx], 0)
        start_time = time()
        for iter_time in range(1000):
            y = torch.prod(tensorx1000[tensor_idx], 0)
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        prod_time[tensor_size] = interval
    return inform, prod_time

def contiguousCopy(size_list):
    tensorx1000 = [torch.rand(20, n, 50) for n in size_list]
    copy_time = {}
    inform  = "\ntest contiguous copy\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = tensorx1000[tensor_idx].clone()
        for iter_time in range(1000):
            y = warmup.clone()
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
    inform  = "\ntest contiguous add\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = tensorx1000[tensor_idx].clone()
        for iter_time in range(1000):
            y = warmup + 5.5
        start_time = time()
        for iter_time in range(1000):
            y = warmup + 5.5 
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        add_time[tensor_size] = interval
    return inform, add_time

def contiguousDiv(size_list):
    tensorx1000 = [torch.randn(20, n, 50) for n in size_list]
    div_time = {}
    inform = "\ntest contiguous div\nsize\t\ttime(us)\n"
    x = torch.div( tensorx1000[-1], 3)
    for tensor_idx in range(len(size_list)):
        warmup = tensorx1000[tensor_idx].clone()
        for iter_time in range(1000):
            y = torch.div(warmup, 3)
        start_time = time()
        for iter_time in range(1000):
            y = torch.div(warmup, 3)
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        div_time[tensor_size] = interval
    return inform, div_time

def contiguousSin(size_list):
    tensorx1000 = [torch.randn(20, n, 50) for n in size_list]
    sin_time = {}
    inform = "\ntest contiguous sin\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = tensorx1000[tensor_idx].clone()
        for iter_time in range(1000):
            y = torch.sin(warmup)
        start_time = time()
        for iter_time in range(1000):
            y = torch.sin(warmup)
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        sin_time[tensor_size] = interval

    return inform, sin_time

def contiguousExp(size_list):
    tensorx1000 = [torch.randn(20, n, 50) for n in size_list]
    exp_time = {}
    inform = "\ntest contiguous exp\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = tensorx1000[tensor_idx].clone()
        for iter_time in range(1000):
            y = torch.exp(warmup)
        start_time = time()
        for iter_time in range(1000):
            y = torch.exp(warmup)
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        exp_time[tensor_size] = interval
    return inform, exp_time

def contiguousSum(size_list):
    tensorx1000 = [torch.randn(20, n, 50) for n in size_list]
    sum_time = {}
    inform = "\ntest contiguous sum\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = tensorx1000[tensor_idx].clone()
        for iter_time in range(1000):
            y = torch.sum(warmup, 0)
        start_time = time()
        for iter_time in range(1000):
            y = torch.sum(warmup, 0)
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        sum_time[tensor_size] = interval
    return inform, sum_time

def contiguousProd(size_list):
    tensorx1000 = [torch.randn(20, n, 50) for n in size_list]
    prod_time = {}
    inform = "\ntest contiguous prod\nsize\t\ttime(us)\n"
    for tensor_idx in range(len(size_list)):
        warmup = tensorx1000[tensor_idx].clone()
        for iter_time in range(1000):
            y = torch.prod(warmup, 0)
        start_time = time()
        for iter_time in range(1000):
            y = torch.prod(warmup, 0)
        end_time = time()
        interval = (end_time - start_time) / 1000
        tensor_size = '%dk' % size_list[tensor_idx]
        prod_time[tensor_size] = interval
    return inform, prod_time

if __name__ == '__main__':
    with open(args.outputfile, 'a') as f:
        if(args.continuity == "contiguous"):
            size_list = size_list[:11]
            if(args.operation == "copy"): 
                copy_inform, copy_time = contiguousCopy(size_list)
                f.write(copy_inform)
                for time in copy_time:
                    f.write(time + '\t\t' + str(copy_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
            elif(args.operation == "add"):
                add_inform, add_time = contiguousAdd(size_list)
                f.write(add_inform)
                for time in add_time:
                    f.write(time + '\t\t' + str(add_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
            elif(args.operation == "div"):
                div_inform, div_time = contiguousDiv(size_list)
                f.write(div_inform)
                for time in div_time:
                    f.write(time + '\t\t' + str(div_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
            elif(args.operation == "exp"):
                exp_inform, exp_time = contiguousExp(size_list)
                f.write(exp_inform)
                for time in exp_time:
                    f.write(time + '\t\t' + str(exp_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
            elif(args.operation == "sin"):
                sin_inform, sin_time = contiguousSin(size_list)
                f.write(sin_inform)
                for time in sin_time:
                    f.write(time + '\t\t' + str(sin_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
            elif(args.operation == "sum"):
                sum_inform, sum_time = contiguousSum(size_list)
                f.write(sum_inform)
                for time in sum_time:
                    f.write(time + '\t\t' + str(sum_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
            elif(args.operation == "prod"):
                prod_inform, prod_time = contiguousProd(size_list)
                f.write(prod_inform)
                for time in prod_time:
                    f.write(time + '\t\t' + str(prod_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
        elif(args.continuity == "discontiguous"):
            if(args.operation == "copy"):
                copy_inform, copy_time = discontigCopy(size_list)
                f.write(copy_inform)
                for time in copy_time:
                    f.write(time + '\t\t' + str(copy_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
            elif(args.operation == "add"):
                add_inform, add_time = discontigAdd(size_list)
                f.write(add_inform)
                for time in add_time:
                    f.write(time + '\t\t' + str(add_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
            elif(args.operation == "div"):
                div_inform, div_time = discontigDiv(size_list)
                f.write(div_inform)
                for time in div_time:
                    f.write(time + '\t\t' + str(div_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
            elif(args.operation == "exp"):
                exp_inform, exp_time = discontigExp(size_list) 
                f.write(exp_inform)
                for time in exp_time:
                    f.write(time + '\t\t' + str(exp_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
            elif(args.operation == "sin"):
                sin_inform, sin_time = discontigSin(size_list)
                f.write(sin_inform)
                for time in sin_time:
                    f.write(time + '\t\t' + str(sin_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
            elif(args.operation == "sum"):
                sum_inform, sum_time = discontigSum(size_list)
                f.write(sum_inform)
                for time in sum_time:
                    f.write(time + '\t\t' + str(sum_time[time]*1e6))
                    f.write('\n')
                f.write('\n')
            elif(args.operation == "prod"):
                prod_inform, prod_time = discontigProd(size_list)
                f.write(prod_inform)
                for time in prod_time:
                    f.write(time + '\t\t' + str(prod_time[time]*1e6))
                    f.write('\n')
                f.write('\n')

