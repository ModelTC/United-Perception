损失函数
========

UP支持以下损失函数：

entropy_loss

  * softmax_cross_entropy

  .. code-block:: yaml

      loss:
        type: softmax_cross_entropy
        kwargs:
          class_dim: -1

  * sigmoid_cross_entropy
  
  .. code-block:: yaml
     
      loss:
        type: sigmoid_cross_entropy

l1_loss
  
  * l1_loss

  .. code-block:: yaml

      loss:
        type: l1_loss
