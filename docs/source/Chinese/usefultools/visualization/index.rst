可视化
======

UP 支持两种可视化模式：

    * Inference
    * Hook

Inference
---------

你可以将可视化设置加入推理配置中，如下所示：

  .. code-block:: yaml
    
    runtime:
      inferencer:
        type: base
        kwargs:
          visualizer:
            type: plt
            kwargs:
              class_names: ['__background__', 'person'] # class names
              thresh: 0.5

Hook
----

UP 支持可视化hook，你可以将可视化设置加入Hook配置中，这样可以在训练和评估时画出 gt 框和检测框。

  .. code-block:: yaml
    
    - type: visualize
      kwargs:
        vis_gt:
          type: plt
          kwargs:
            output_dir: vis_gt
            thresh: 0.3
        vis_dt:
          type: plt
          kwargs:
            output_dir: vis_dt
            thresh: 0.3


  
