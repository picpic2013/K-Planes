import torch

if __name__ == '__main__':
    a = torch.arange(10).repeat(10)
    b = torch.ones(10, requires_grad=True, dtype=torch.float32)

    aa = a.long()
    c = b[aa]

    d = c.sum()

    d.backward()

    print(b.grad)