# register_ply

## brief

`register_ply` processes pointcloud registration in sim(3) space, and it has main two stage:

1. Find **iss** and **FPFH** features for initialization of transformation
2. Refine transformation based on manifold optimization in **sim(3)** by reducing nearest point to point distance


## install

```
git clone https://github.com/tanzby/register_ply.git --depth 1
cd register_ply
mkdir build && cd build 
cmake .. && make -j4
```

## usage

```
./build/cloud_registration source.ply target.ply
```


## todos

- [ ] add configuration settings
- [ ] support pcd file
- [ ] add point to plane factor, or other robust residual factors



## reference

1. https://www.freesion.com/article/2404870876/
