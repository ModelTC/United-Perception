## baseline
| model                   | backbone        | anchor | dataset | bs | epoch | 3d_AP Car/Pedestrian/Cyclist| model |
| ----------------------- | --------------- | --- | ----- | -- | --- | ----------------------------- | ----- |
| [pointpillar](https://github.com/ModelTC/EOD/tree/main/configs/det_3d/pointpillar/pointpillar.yaml)       | PillarVFE            |   yes     | kitti  | 32 | 80      | 76.99/49.40/63.35 | [ckpt](http://spring.sensetime.com/dropadmin/$/o4Ass.pth)  |
| [second](https://github.com/ModelTC/EOD/tree/main/configs/det_3d/second/second.yaml)| VoxelBackBone8x           |   yes     | kitti  | 32 | 80     | 78.65/53.76/64.23 |    [ckpt](http://spring.sensetime.com/dropadmin/$/tDFO4.pth)   |
| [centerpoint_pillar](https://github.com/ModelTC/EOD/tree/main/configs/det_3d/centerpoint/centerpoint_pillar.yaml)| PillarVFE            |    no    | kitti  | 32 | 80     | 75.00/51.10/60.12     |  [ckpt](http://spring.sensetime.com/dropadmin/$/7DwKD.pth)   |
| [centerpoint_second](https://github.com/ModelTC/EOD/tree/main/configs/det_3d/centerpoint/centerpoint_second.yaml) | VoxelBackBone8x            |    no    | kitti  | 32 | 80     | 77.28/54.31/68.20  | [ckpt](http://spring.sensetime.com/dropadmin/$/QGGC7.pth)  |
