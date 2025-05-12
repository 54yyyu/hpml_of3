import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb

# ─── Model Definition ───────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1    = nn.Conv2d(1, 32, 3, 1)
        self.conv2    = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1      = nn.Linear(9216, 128)
        self.fc2      = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ─── Training (performance only) ────────────────────────────────────────────────
def train_perf(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    # Reset and start timing & memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()
    else:
        t0 = time.time()

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _ = model(data)       # forward
        # we skip loss/backward—only measuring forward pass performance here
        optimizer.step()      # dummy step to simulate full iteration

    # end timing
    if device.type == 'cuda':
        end_evt.record()
        torch.cuda.synchronize()
        train_ms = start_evt.elapsed_time(end_evt)
        peak_mb  = torch.cuda.max_memory_reserved() / (1024**2)
    else:
        train_ms = (time.time() - t0) * 1000.0
        peak_mb  = 0.0

    # Log to W&B
    wandb.log({
        "epoch": epoch,
        "train_time_ms": train_ms,
        "train_peak_mem_mb": peak_mb
    })

    print(f"[Epoch {epoch}] train_time: {train_ms:.1f} ms, peak_mem: {peak_mb:.1f} MB")

# ─── Testing (performance only) ─────────────────────────────────────────────────
def test_perf(model, device, test_loader, epoch):
    model.eval()
    # Reset and start timing & memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()
    else:
        t0 = time.time()

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            _ = model(data)

    # end timing
    if device.type == 'cuda':
        end_evt.record()
        torch.cuda.synchronize()
        test_ms = start_evt.elapsed_time(end_evt)
        peak_mb = torch.cuda.max_memory_reserved() / (1024**2)
    else:
        test_ms = (time.time() - t0) * 1000.0
        peak_mb = 0.0

    # Log to W&B
    wandb.log({
        "epoch": epoch,
        "test_time_ms": test_ms,
        "test_peak_mem_mb": peak_mb
    })

    print(f"[Epoch {epoch}] test_time: {test_ms:.1f} ms, peak_mem: {peak_mb:.1f} MB")

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='MNIST Perf Profile (ReLU)')
    parser.add_argument('--batch-size',      type=int,   default=64)
    parser.add_argument('--test-batch-size', type=int,   default=1000)
    parser.add_argument('--epochs',          type=int,   default=5)
    parser.add_argument('--lr',              type=float, default=1.0)
    parser.add_argument('--gamma',           type=float, default=0.7)
    parser.add_argument('--no-cuda',         action='store_true')
    parser.add_argument('--no-mps',          action='store_true')
    parser.add_argument('--seed',            type=int,   default=1)
    parser.add_argument('--log-interval',    type=int,   default=10)
    parser.add_argument('--save-model',      action='store_true')
    args = parser.parse_args()

    # Initialize W&B (only performance metrics)
    wandb.init(
        project="mnist-perf-relu",
        config=vars(args),
        save_code=True
    )
    cfg = wandb.config

    use_cuda = not cfg['no_cuda'] and torch.cuda.is_available()
    use_mps  = not cfg['no_mps'] and torch.backends.mps.is_available()
    torch.manual_seed(cfg['seed'])

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Data loaders
    train_kwargs = {'batch_size': cfg['batch_size'], 'shuffle': True}
    test_kwargs  = {'batch_size': cfg['test_batch_size'], 'shuffle': False}
    if use_cuda:
        train_kwargs.update({'num_workers': 1, 'pin_memory': True})
        test_kwargs .update({'num_workers': 1, 'pin_memory': True})

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform),
                              **train_kwargs)
    test_loader  = DataLoader(datasets.MNIST('../data', train=False, transform=transform),
                              **test_kwargs)

    # Model & optimizer
    model     = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=cfg['lr'])
    scheduler = StepLR(optimizer, step_size=1, gamma=cfg['gamma'])

    # Perf loop
    for epoch in range(1, cfg['epochs'] + 1):
        train_perf(model, device, train_loader, optimizer, epoch, cfg['log_interval'])
        test_perf(model, device, test_loader, epoch)
        scheduler.step()

        if cfg['save_model']:
            torch.save(model.state_dict(), "mnist_cnn_perf.pt")

    wandb.finish()

if __name__ == '__main__':
    main()

