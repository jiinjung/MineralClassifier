"""
Evaluation metrics for semantic segmentation.

This module provides pixel-level accuracy and Intersection over Union (IoU)
metrics for evaluating segmentation model performance.
"""

import torch


def pixel_accuracy(predictions, target):
    """
    Compute pixel-wise accuracy.
    
    Args:
        predictions: Predicted class labels, shape [B, H, W] or [H, W]
        target: Ground truth labels, shape [B, H, W] or [H, W]
        
    Returns:
        Pixel accuracy as a float (0.0 to 1.0)
    """
    correct = (predictions == target).float()
    pa = correct.sum() / target.numel()
    return pa


def compute_iou(prediction, target, num_classes):
    """
    Compute Intersection over Union (IoU) for each class.
    
    Args:
        prediction: Predicted class labels, shape [B, H, W] or [H, W]
        target: Ground truth labels, shape [B, H, W] or [H, W]
        num_classes: Number of classes
        
    Returns:
        List of IoU values for each class. Returns NaN for classes
        that don't appear in either prediction or target.
    """
    ious = []
    prediction = prediction.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = prediction == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().float()
        union = (pred_inds.long().sum().float() + target_inds.long().sum().float()) - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(union))
            
    return ious