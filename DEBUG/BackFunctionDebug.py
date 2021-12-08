import torch


def DebugTopK():
    x =torch.tensor([1., 4., 2.], requires_grad = True)

    print(x.requires_grad)
    y = torch.topk(x, 3)
    print("y:", y.values.requires_grad)


if __name__ == '__main__':
    DebugTopK()
