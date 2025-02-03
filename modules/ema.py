import torch
import torch.distributed as dist


class EfficientEMA:
    def __init__(self, model, decay, device, sync_interval=100):
        self.model = model  # the EMA model (teacher)
        self.decay = decay  # 0.999
        self.device = device
        self.shadow_params = {}  # Dictionary to store EMA parameters
        self.shadow_buffers = {}
        self.sync_interval = sync_interval
        self.update_counter = 0
        for p in self.model.parameters():
            p.requires_grad_(False)
    
    @staticmethod
    def _named_parameters(model):
        return model.module.named_parameters() if hasattr(model, 'module') else model.named_parameters()
    
    @staticmethod
    def _named_buffers(model):
        return model.module.named_buffers() if hasattr(model, 'module') else model.named_buffers()        

    def register(self):
        """Initialize the shadow parameters with current (teacher) model."""
        for name, param in self.model.named_parameters():
            self.shadow_params[name] = param.data.clone()
        for name, buf in self.model.named_buffers():
            self.shadow_buffers[name] = buf.data.clone()

    def update(self, model):
        """Update shadow parameters from external (student) model."""
        if self.decay < 1:
            with torch.no_grad():
                for name, param in self._named_parameters(model):
                    assert name in self.shadow_params, f'Parameter name mismatch: {name}'
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow_params[name]
                    self.shadow_params[name].copy_(new_average)
                
                for name, buf in self._named_buffers(model):
                    assert name in self.shadow_buffers, f'Buffer name mismatch: {name}'
                    new_average = (1.0 - self.decay) * buf.data + self.decay * self.shadow_buffers[name]
                    self.shadow_buffers[name].copy_(new_average)

        self.update_counter += 1
        if self.update_counter % self.sync_interval == 0:
            self.synchronize()

    def synchronize(self):
        """Synchronize the shadow parameters across all processes."""
        with torch.no_grad():
            for name in self.shadow_params.keys():
                dist.all_reduce(self.shadow_params[name], op=dist.ReduceOp.AVG)
            
            for name in self.shadow_buffers.keys():
                dist.all_reduce(self.shadow_buffers[name], op=dist.ReduceOp.AVG)

    def apply_shadow(self):
        """Apply the shadow parameters to the EMA (teacher) model."""
        for name, param in self.model.named_parameters():
            assert name in self.shadow_params, f'Parameter name mismatch: {name}'
            param.data.copy_(self.shadow_params[name])
            
        for name, buf in self.model.named_buffers():
            assert name in self.shadow_buffers, f'Buffer name mismatch: {name}'
            buf.data.copy_(self.shadow_buffers[name])

# # In the main training loop:
# ema = EfficientEMA(ema_model, decay=0.999, device=device, sync_interval=100)
# ema.register()

# for epoch in range(num_epochs):
#     for batch in dataloader:
#         # ... (training steps) ...
        
#         # Update EMA
#         ema.update()

# # Use EMA model for inference
# ema.apply_shadow()
# # Perform inference with ema_model