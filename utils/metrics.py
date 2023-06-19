# pixel accuracy
def pixel_accuracy(predictions, target):
    correct = (predictions == target).float()
    pa = correct.sum() / target.numel()
    return pa


# Intersection over union
def compute_iou(prediction, target, num_classes):
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
            ious.append(float(intersection) / float(union))  # float division
            
    return ious
