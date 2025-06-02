# PlanTF 学习记录

## 模型结构

### MLP位置编码

Linear(4, 128)

Linear(128, 128)

> 归一化层自己已经能控制输出的偏移和平移了，再加上前一层的 bias 会变得冗余。
## 模型特征
### ego_feature
| key | shape | type |
| :----: | :-----: | :-----: |
| position |  (T, 2) | float64 |
| heading  | T | float64 |
| velocity  | (T, 2) | float64 |
| acceleration  | (T, 2) | float64 |
| shape  | (T, 2) | float64 |
| category  | -1 | int8 |
| valid_mask  | T | bool8 |

### agent_feature
| key | shape | type |
| :----: | :-----: | :-----: |
| position |  (N, T, 2) | float64 |
| heading  | (N, T) | float64 |
| velocity  | (N, T, 2) | float64 |
| shape  |(N, T, 2) | float64 |
| category  | (N,) | int8 |
| valid_mask  | (N, T) | bool8 |

### map_feature

| key  | shape | type | 说明 |
| :----: | :-----: | :-----: | :------: |
| point_position | (M, 3, P, 2) | float64 |  |
| point_vector | (M, 3, P, 2) |  |  |
| point_side |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

 

​        point_position = np.zeros((M, 3, P, 2), dtype=np.float64)

​        point_vector = np.zeros((M, 3, P, 2), dtype=np.float64)

​        point_side = np.zeros((M, 3), dtype=np.int8)

​        point_orientation = np.zeros((M, 3, P), dtype=np.float64)

​        polygon_center = np.zeros((M, 3), dtype=np.float64)

​        polygon_position = np.zeros((M, 2), dtype=np.float64)

​        polygon_orientation = np.zeros(M, dtype=np.float64)

​        polygon_type = np.zeros(M, dtype=np.int8)

​        polygon_on_route = np.zeros(M, dtype=np.bool)

​        polygon_tl_status = np.zeros(M, dtype=np.int8)

​        polygon_speed_limit = np.zeros(M, dtype=np.float64)

​        polygon_has_speed_limit = np.zeros(M, dtype=np.bool)
