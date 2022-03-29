Loss function
=============

UP supports the following loss functions.

entropy_loss

  * softmax_cross_entropy

  .. code-block::

      loss:
        type: softmax_cross_entropy
        kwargs:
          class_dim: -1

  * sigmoid_cross_entropy
  
  .. code-block::
     
      loss:
        type: sigmoid_cross_entropy

l1_loss
  
  * l1_loss

  .. code-block::

      loss:
        type: l1_loss
