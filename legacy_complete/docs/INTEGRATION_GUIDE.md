# PyTorch BFO Optimizer Integration Guide

## Table of Contents
1. [Framework Integration](#framework-integration)
2. [PyTorch Lightning](#pytorch-lightning)
3. [Hugging Face Transformers](#hugging-face-transformers)
4. [FastAI](#fastai)
5. [Distributed Training](#distributed-training)
6. [Custom Training Loops](#custom-training-loops)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Production Deployment](#production-deployment)

## Framework Integration

### PyTorch Lightning

PyTorch Lightning integration requires custom optimizer configuration:

```python
import pytorch_lightning as pl
from pytorch_bfo_optimizer import BFO, HybridBFO

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        # Use HybridBFO for gradient compatibility
        optimizer = HybridBFO(
            self.parameters(),
            population_size=20,
            gradient_weight=0.5
        )
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Custom closure for BFO
        def closure():
            self.zero_grad()
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            self.manual_backward(loss)
            return loss.item()
        
        # Manual optimization
        opt = self.optimizers()
        loss_val = opt.step(closure)
        
        # Logging
        self.log('train_loss', loss_val)
        return {'loss': loss_val}
    
    def configure_callbacks(self):
        # Custom callback for BFO monitoring
        return [BFOCallback()]

class BFOCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Access optimizer state
        opt = trainer.optimizers[0]
        if hasattr(opt, 'best_fitness'):
            trainer.logger.log_metrics({
                'best_fitness': opt.best_fitness,
                'population_diversity': opt._compute_diversity()
            }, step=trainer.global_step)

# Training
model = LitModel()
trainer = pl.Trainer(
    max_epochs=10,
    automatic_optimization=False  # Required for BFO
)
trainer.fit(model, train_dataloader)
```

### Hugging Face Transformers

Integration with Transformers requires custom trainer:

```python
from transformers import Trainer, TrainingArguments
from pytorch_bfo_optimizer import AdaptiveBFO
import torch

class BFOTrainer(Trainer):
    def create_optimizer(self):
        """Override to use BFO optimizer."""
        if self.optimizer is None:
            self.optimizer = AdaptiveBFO(
                self.model.parameters(),
                population_size=30,
                adaptation_rate=0.1
            )
        return self.optimizer
    
    def training_step(self, model, inputs):
        """Custom training step for BFO."""
        model.train()
        
        def closure():
            self.optimizer.zero_grad()
            loss = self.compute_loss(model, inputs)
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
            return loss.item()
        
        # Step optimizer
        tr_loss = self.optimizer.step(closure)
        
        return tr_loss

# Usage
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_steps=10,
)

trainer = BFOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### FastAI

FastAI integration using custom optimizer:

```python
from fastai.vision.all import *
from pytorch_bfo_optimizer import HybridBFO

# Custom optimizer function
def bfo_opt(params, lr=0.1, **kwargs):
    return HybridBFO(params, population_size=20, gradient_weight=0.5)

# Create learner with BFO
learn = cnn_learner(
    dls, 
    resnet34, 
    opt_func=bfo_opt,
    metrics=accuracy
)

# Custom training loop for BFO
@patch
def fit_bfo(self: Learner, n_epoch, lr=None, **kwargs):
    self.opt = self.opt_func(self.model.parameters())
    
    for epoch in range(n_epoch):
        for batch in self.dls.train:
            def closure():
                self.model.zero_grad()
                loss = self.loss_func(self.model(*batch[:-1]), *batch[-1:])
                loss.backward()
                return loss.item()
            
            loss_val = self.opt.step(closure)
            self.recorder.log(loss_val)

# Train
learn.fit_bfo(10)
```

## Distributed Training

### Data Parallel (DP)

```python
import torch.nn as nn
from pytorch_bfo_optimizer import BFO

# Wrap model in DataParallel
model = nn.DataParallel(model)
optimizer = BFO(model.parameters(), population_size=50)

# Training remains the same
def closure():
    output = model(data)
    loss = criterion(output, target)
    return loss.item()

optimizer.step(closure)
```

### Distributed Data Parallel (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # BFO with DDP considerations
    optimizer = BFO(
        ddp_model.parameters(),
        population_size=50 // world_size,  # Divide population
        device=f'cuda:{rank}'
    )
    
    # Synchronize fitness across processes
    def closure():
        output = ddp_model(data)
        loss = criterion(output, target)
        
        # All-reduce loss across processes
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= world_size
        
        return loss.item()
    
    # Training loop
    for epoch in range(num_epochs):
        loss = optimizer.step(closure)
        
        # Synchronize best parameters
        if rank == 0:
            # Broadcast best parameters from rank 0
            for param in ddp_model.parameters():
                dist.broadcast(param.data, src=0)

# Launch
mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)
```

## Custom Training Loops

### Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

model = model.cuda()
optimizer = HybridBFO(model.parameters())
scaler = GradScaler()

def amp_closure():
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Scale loss and compute gradients
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    
    # Unscale for BFO
    scaler.unscale_(optimizer)
    
    return loss.item()

# Training step
loss = optimizer.step(amp_closure)
scaler.update()
```

### Gradient Accumulation

```python
accumulation_steps = 4
optimizer = BFO(model.parameters())

def accumulated_closure():
    total_loss = 0
    
    for i in range(accumulation_steps):
        # Get mini-batch
        data, target = next(data_iter)
        
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target) / accumulation_steps
            total_loss += loss.item()
    
    return total_loss

# Step every accumulation_steps
if step % accumulation_steps == 0:
    loss = optimizer.step(accumulated_closure)
```

### Learning Rate Scheduling

```python
class BFOScheduler:
    def __init__(self, optimizer, schedule_fn):
        self.optimizer = optimizer
        self.schedule_fn = schedule_fn
        self.epoch = 0
    
    def step(self):
        # Update BFO parameters
        params = self.schedule_fn(self.epoch)
        for key, value in params.items():
            if key in self.optimizer.defaults:
                self.optimizer.defaults[key] = value
        self.epoch += 1

# Define schedule
def schedule(epoch):
    return {
        'step_size_max': 0.1 * (0.95 ** epoch),
        'elim_prob': min(0.5, 0.25 * (1.05 ** epoch))
    }

scheduler = BFOScheduler(optimizer, schedule)

# Use in training
for epoch in range(num_epochs):
    train_epoch()
    scheduler.step()
```

## Monitoring and Logging

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/bfo_experiment')

class MonitoredBFO(BFO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0
    
    def step(self, closure):
        # Regular step
        loss = super().step(closure)
        
        # Log metrics
        writer.add_scalar('Loss/train', loss, self.step_count)
        writer.add_scalar('BFO/best_fitness', self.best_fitness, self.step_count)
        writer.add_scalar('BFO/population_diversity', 
                         self._compute_diversity(), self.step_count)
        
        # Log population distribution
        if self.step_count % 100 == 0:
            writer.add_histogram('BFO/population', 
                               self.population.flatten(), self.step_count)
        
        self.step_count += 1
        return loss

# Use monitored optimizer
optimizer = MonitoredBFO(model.parameters())
```

### Weights & Biases Integration

```python
import wandb

wandb.init(project="bfo-optimization", config={
    "population_size": 50,
    "learning_rate": 0.1,
    "architecture": "resnet18",
})

class WandBBFO(AdaptiveBFO):
    def step(self, closure):
        loss = super().step(closure)
        
        # Log to W&B
        wandb.log({
            "loss": loss,
            "best_fitness": self.best_fitness,
            "population_diversity": self._compute_diversity(),
            "step_size_max": self.defaults["step_size_max"],
            "elimination_prob": self.defaults["elim_prob"],
        })
        
        return loss

# Track model
wandb.watch(model)
optimizer = WandBBFO(model.parameters())
```

### Custom Metrics Tracking

```python
class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'losses': [],
            'diversities': [],
            'step_sizes': [],
            'convergence_rate': []
        }
    
    def update(self, optimizer, loss):
        self.metrics['losses'].append(loss)
        self.metrics['diversities'].append(optimizer._compute_diversity())
        self.metrics['step_sizes'].append(optimizer.defaults['step_size_max'])
        
        # Calculate convergence rate
        if len(self.metrics['losses']) > 10:
            recent = self.metrics['losses'][-10:]
            rate = (recent[0] - recent[-1]) / (recent[0] + 1e-8)
            self.metrics['convergence_rate'].append(rate)
    
    def plot_metrics(self):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0,0].plot(self.metrics['losses'])
        axes[0,0].set_title('Loss')
        
        axes[0,1].plot(self.metrics['diversities'])
        axes[0,1].set_title('Population Diversity')
        
        axes[1,0].plot(self.metrics['step_sizes'])
        axes[1,0].set_title('Step Size')
        
        axes[1,1].plot(self.metrics['convergence_rate'])
        axes[1,1].set_title('Convergence Rate')
        
        plt.tight_layout()
        plt.show()

# Usage
tracker = MetricsTracker()
for step in range(1000):
    loss = optimizer.step(closure)
    tracker.update(optimizer, loss)

tracker.plot_metrics()
```

## Production Deployment

### Model Serving

```python
# Save trained model with BFO metadata
def save_bfo_model(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_config': {
            'population_size': optimizer.defaults['population_size'],
            'best_fitness': optimizer.best_fitness,
            'best_params': optimizer.best_params
        }
    }, path)

# Load for inference
def load_bfo_model(model_class, path):
    checkpoint = torch.load(path)
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['optimizer_config']

# Inference server
from flask import Flask, request, jsonify

app = Flask(__name__)
model, config = load_bfo_model(MyModel, 'model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    data = torch.tensor(request.json['data'])
    with torch.no_grad():
        output = model(data)
    return jsonify({
        'prediction': output.tolist(),
        'model_fitness': config['best_fitness']
    })
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install pytorch-bfo-optimizer

# Copy model and code
COPY model.pth .
COPY app.py .

# Run server
CMD ["python", "app.py"]
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bfo-model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bfo-model
  template:
    metadata:
      labels:
        app: bfo-model
    spec:
      containers:
      - name: model-server
        image: myregistry/bfo-model:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        ports:
        - containerPort: 5000
---
apiVersion: v1
kind: Service
metadata:
  name: bfo-model-service
spec:
  selector:
    app: bfo-model
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

### Performance Optimization for Production

```python
# Optimized inference configuration
class ProductionBFOModel:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        
        # Compile for inference
        if device == 'cuda':
            self.model = torch.compile(self.model, mode='reduce-overhead')
        
        # Warm up
        self.warmup()
    
    def load_model(self, path):
        model = MyModel()
        model.load_state_dict(torch.load(path)['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def warmup(self, num_runs=10):
        dummy_input = torch.randn(1, 784).to(self.device)
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(dummy_input)
    
    @torch.no_grad()
    def predict(self, x):
        x = x.to(self.device)
        return self.model(x).cpu()
    
    def batch_predict(self, batch, max_batch_size=128):
        results = []
        for i in range(0, len(batch), max_batch_size):
            sub_batch = batch[i:i+max_batch_size]
            results.append(self.predict(sub_batch))
        return torch.cat(results)
```

## Best Practices for Integration

1. **Choose the Right Variant**: Use HybridBFO for framework integration
2. **Handle Closures Properly**: Ensure closures return scalar values
3. **Monitor Population Health**: Track diversity and convergence
4. **Scale Appropriately**: Adjust population size for distributed settings
5. **Profile Performance**: Use framework-specific profiling tools
6. **Version Control**: Track optimizer hyperparameters with models
7. **Test Thoroughly**: Validate integration with small examples first

## Troubleshooting Integration Issues

### Common Framework Issues

1. **Automatic Mixed Precision**: May require custom handling
2. **Distributed Training**: Population synchronization needed
3. **Gradient Accumulation**: Adjust closure for accumulated gradients
4. **Learning Rate Schedulers**: Adapt for BFO parameters

### Performance Considerations

1. **Batch Size**: Larger batches generally better for BFO
2. **Population Size**: May need reduction for memory constraints
3. **Compilation**: Test torch.compile compatibility
4. **Profiling**: Use framework profilers to identify bottlenecks

## Conclusion

BFO can be integrated with most PyTorch-based frameworks with appropriate modifications. The key is understanding how the framework handles optimization and adapting the closure function accordingly. For production deployment, focus on inference optimization and proper model serialization.