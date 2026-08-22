import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(3000, 3000, device=device)
y = torch.randn(3000, 3000, device=device)

z = x @ y

print("Calculation finished!")
print(z.shape)