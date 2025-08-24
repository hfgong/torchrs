class Task:
    """Base class for recommendation tasks.
    
    This class defines the interface for tasks such as retrieval and ranking
    in recommendation systems. Subclasses should implement task-specific
    loss computation and metrics.
    
    Tasks encapsulate the training objective and evaluation metrics for
    specific recommendation scenarios. For example, a retrieval task might
    use sampled softmax loss and FactorizedTopK metrics, while a ranking
    task might use mean squared error loss and RMSE metrics.
    
    The task abstraction allows for modular design where the same model
    architecture can be trained for different objectives by swapping tasks.
    """
    
    def __init__(self):
        """Initialize the base task.
        
        The base task constructor is minimal since most task-specific
        configuration happens in subclasses. This ensures a clean
        interface for different types of recommendation tasks.
        """
        pass
        
    def compute_loss(self, predictions, targets):
        """Compute the loss for this task.
        
        This method should be overridden by subclasses to implement
        task-specific loss computation. The loss function defines
        what the model optimizes during training.
        
        For retrieval tasks, this might be sampled softmax loss.
        For ranking tasks, this might be mean squared error or cross-entropy.
        
        Args:
            predictions: Model predictions (e.g., embeddings, scores)
            targets: Ground truth targets (e.g., labels, ratings)
            
        Returns:
            Computed loss value as a tensor
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("compute_loss method must be implemented by subclasses")
        
    def compute_metrics(self, predictions, targets):
        """Compute metrics for this task.
        
        This method should be overridden by subclasses to implement
        task-specific metric computation. Metrics are used to evaluate
        model performance during training and testing.
        
        For retrieval tasks, this might include recall, precision, or NDCG.
        For ranking tasks, this might include RMSE, MAE, or ranking-specific metrics.
        
        Args:
            predictions: Model predictions (e.g., embeddings, scores)
            targets: Ground truth targets (e.g., labels, ratings)
            
        Returns:
            Dictionary of computed metrics (metric_name -> value)
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("compute_metrics method must be implemented by subclasses")