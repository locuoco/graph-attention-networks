from keras import losses
from keras import ops
from keras import backend

class MeanAbsoluteScaledError(losses.Loss):
	def __init__(self, reduction='sum_over_batch_size', name='mean_absolute_scaled_error'):
		super().__init__(reduction=reduction, name=name)

	def call(self, y_true, y_pred):
		epsilon = backend.epsilon()
		y_true = ops.reshape(y_true, (-1, y_true.shape[-1]))
		y_pred = ops.reshape(y_pred, (-1, y_pred.shape[-1]))
		sae = ops.sum(ops.abs(y_true - y_pred), axis=0)
		sad = ops.sum(ops.abs(y_true - ops.mean(y_true, axis=0)), axis=0)
		return sae / ops.maximum(sad, epsilon)






