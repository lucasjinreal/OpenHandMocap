# OpenHandMocap

Gathering some SOTA hand mocap models. Code under constructions. Eventually we will make a **standard, ready for production** version to do hand mocap along with body.

Here is the result.


![](https://s1.ax1x.com/2022/08/26/v2TtUO.png)
![](https://s1.ax1x.com/2022/08/26/v2Td8H.png)
![](https://github.com/jinfagang/public_images/raw/master/a.gif)


More results are coming.

We provide 2 kinds of hand mocap model:

- With a hand box detector: more accurate;
- Without hand box detector, but need bodymocap, using bodymocap result to get hand box, less accurate but more pratical.

Except `ohamo`, other folders are **vendor** models with our modifications. `**ohamo**` is our *OpenHandMocap System*.


`ohamo` aiming at providing a **fast, accurate, out-of-box** 3D hand mocap system **with training code**. (`ohamo` itself is fully deployed running via ONNX doesn't need pytorch.)

## Quick Start

`ohamo` is our pipeline, you don't need any prerequests before running, just install some package that you might don't have:

- alfred-py

Then, just run:

```
python main.py -i ~/Videoes/a.mp4
```

You will get a result on hand mocap.

> Be note: Our hand detection model running realtime even on CPU, it's inference is on ONNXRuntime on CPU by default. For further accerlate please using TensorRT.



## Model Supported

1. `frankhand`:

frankhand is a simpified version from frankmocap, we edited the pipeline to a hand detector free style, you can get hand without using hand detector;

2. `mobrecon`:

this is a decent hand mocap model, but code like a shit and messy. I simplifed it and make it more easy to inference. 



## RoadMap

- We will release a advanced hand detector model for easily setup hand detector;
- We will integrate with body mocap in future;



## Install

- torch
- open3d
- alfred-py

## Models

models supported:

- minimalhand;
- mobrecon;





## References

1. [mobileheand](https://github.com/gmntu/mobilehand)
