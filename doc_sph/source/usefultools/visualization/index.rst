可视化/Visualization
====================

UP 支持两种可视化模式：

    * Inference
    * Hook

推理/Inference
--------------

你可以将可视化设置加入推理配置中，如下所示：

  .. code-block:: yaml
    
    inference:
      visualizer:
        type: plt
        kwargs:
          class_names: ['__background__', 'person'] # class names
          thresh: 0.5

运行时钩子/Hook
---------------

UP 支持可视化钩子，你可以将可视化设置加入钩子配置中，这样可以在训练和评估时画出 gt 框和检测框。

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


  