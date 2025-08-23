class Task:
    """Base class for recommendation tasks.
    
    This class defines the interface for tasks such as retrieval and ranking
    in recommendation systems. Subclasses should implement task-specific
    loss computation and metrics.
    """
    
    def __init__(self):
        pass
        
    def compute_loss(self, predictions, targets):
        """Compute the loss for this task.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Computed loss value
        """
        raise NotImplementedError("compute_loss method must be implemented by subclasses")
        
    def compute_metrics(self, predictions, targets):
        """Compute metrics for this task.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of computed metrics
        """
        raise NotImplementedError("compute_metrics method must be implemented by subclasses")