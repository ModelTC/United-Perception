# Visualization
EOD supports two visualization modes:
* Inference
* Hook

## Inference
You could add visualizer settings into inference config, refering to [Demo](../README.md):

```yaml
inference:
  visualizer:
    type: plt
    kwargs:
      class_names: ['__background__', 'person'] # class names
      thresh: 0.5
```

## Hook
EOD supports **Visualize** hook, you could add visualizer settings into hook config so that drawing gt bboxes and dt results during training and evaluating.

```yaml
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
```
