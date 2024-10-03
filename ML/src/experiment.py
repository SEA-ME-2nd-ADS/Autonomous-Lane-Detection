from model.clrernet import CLRerNet
import torch
import time

def main():
    model = CLRerNet()
    exp = torch.rand(16, 3, 800, 400)
    start_time = time.time()
    output = model(exp)
    end_time = time.time()
    print("time: ", end_time - start_time)
    print("output length: " , len(output))
    for k in output[0]:
        print(k, ": ", output[0][k].shape)

if __name__ == '__main__':
    main()