Visualization
=============

UP supports two modes of visualization.

    * Inference
    * Hook

Inference
---------

You can add the visualization setting into the inference setting as followed.

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

UP supports visualized hook. You can add the visualization setting into the hook setting for drawing gt and dt boxes in training and evaluting.

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


  
