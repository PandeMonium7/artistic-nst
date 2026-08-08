# fast-neural-style

This repository contains Torch code for real-time style transfer and super-resolution using perceptual loss functions.

Feedforward neural networks are trained to apply artistic styles to images. After training, these feedforward networks can stylize images **hundreds of times faster** than optimization-based style transfer methods.

This repository also includes an implementation of instance normalization. Replacing batch normalization with instance normalization significantly improves the visual quality and computational efficiency of feedforward style transfer models.

Stylizing an image at a resolution of 1200x630 takes **50 milliseconds** on a Pascal Titan X:

<div align='center'>
  <img src='images/styles/candy.jpg' height="225px">
  <img src='images/content/hoovertowernight.jpg' height="225px">
  <img src='images/outputs/hoovertowernight_candy.jpg' height="346px">
</div>

In this repository we provide:
- Pretrained feedforward style transfer models ([`models/eccv16`](#baseline-pretrained-models))
- Additional models [using instance normalization](#models-with-instance-normalization)
- Code for [running models on new images](#running-on-new-images)
- A demo that runs models in [real-time off a webcam](#webcam-demo)
- Code for [training new feedforward style transfer models](doc/training.md)
- An implementation of [optimization-based style transfer](#optimization-based-style-transfer)

## Setup
All code is implemented in [Torch](http://torch.ch/).

First [install Torch](http://torch.ch/docs/getting-started.html#installing-torch), then
update / install the following packages:

```bash
luarocks install torch
luarocks install nn
luarocks install image
luarocks install lua-cjson
```

### (Optional) GPU Acceleration

If you have an NVIDIA GPU, you can accelerate all operations with CUDA.

First [install CUDA](https://developer.nvidia.com/cuda-downloads), then
update / install the following packages:

```bash
luarocks install cutorch
luarocks install cunn
```

### (Optional) cuDNN

When using CUDA, you can use cuDNN to accelerate convolutions.

First [download cuDNN](https://developer.nvidia.com/cudnn) and copy the
libraries to `/usr/local/cuda/lib64/`. Then install the Torch bindings for cuDNN:

```bash
luarocks install cudnn
```

### Pretrained Models
Download all pretrained style transfer models by running the script

```bash
bash models/download_style_transfer_models.sh
```

This will download model files (~200MB) to the folder `models/`.

## Baseline Pretrained Models
The baseline style transfer models are located in the folder `models/eccv16`.
Here are some example results where we use these models to stylize an
image of the Chicago skyline at an image size of 512:

<div align='center'>
  <img src='images/content/chicago.jpg' height="185px">
</div>
<img src='images/styles/starry_night_crop.jpg' height="155px">
<img src='images/styles/la_muse.jpg' height="155px">
<img src='images/styles/composition_vii.jpg' height='155px'>
<img src='images/styles/wave_crop.jpg' height='155px'>
<br>
<img src='images/outputs/eccv16/chicago_starry_night.jpg' height="142px">
<img src='images/outputs/eccv16/chicago_la_muse.jpg' height="142px">
<img src='images/outputs/eccv16/chicago_composition_vii.jpg' height="142px">
<img src='images/outputs/eccv16/chicago_wave.jpg' height="142px">

## Models with instance normalization
Replacing batch normalization with instance normalization significantly improves the quality
of feedforward style transfer models.

Models trained with instance normalization are located in the folder `models/instance_norm`.

These models use the same general architecture as the baseline feedforward models, except with
half the number of filters per layer and with instance normalization instead of
batch normalization. Using narrower layers makes the models smaller and faster
without sacrificing model quality.

Here are some example outputs from these models, with an image size of 1024:

<div align='center'>
  <img src='images/styles/candy.jpg' height='174px'>
  <img src='images/outputs/chicago_candy.jpg' height="174px">
  <img src='images/outputs/chicago_udnie.jpg' height="174px">
  <img src='images/styles/udnie.jpg' height='174px'>
  <br>
  <img src='images/styles/the_scream.jpg' height='174px'>
  <img src='images/outputs/chicago_scream.jpg' height="174px">
  <img src='images/outputs/chicago_mosaic.jpg' height="174px">
  <img src='images/styles/mosaic.jpg' height='174px'>
  <br>
  <img src='images/styles/feathers.jpg' height='173px'>
  <img src='images/outputs/chicago_feathers.jpg' height="173px">
  <img src='images/outputs/chicago_muse.jpg' height="173px">
  <img src='images/styles/la_muse.jpg' height='173px'>
</div>

## Running on new images
The script `fast_neural_style.lua` lets you use a trained model to stylize new images:

```bash
th fast_neural_style.lua \
  -model models/eccv16/starry_night.t7 \
  -input_image images/content/chicago.jpg \
  -output_image out.png
```

You can run the same model on an entire directory of images like this:

```bash
th fast_neural_style.lua \
  -model models/eccv16/starry_night.t7 \
  -input_dir images/content/ \
  -output_dir out/
```

You can control the size of the output images using the `-image_size` flag.

By default this script runs on CPU; to run on GPU, add the flag `-gpu`
specifying the GPU on which to run.

The full set of options for this script is [described here](doc/flags.md#fast_neural_stylelua).


## Webcam demo
You can use the script `webcam_demo.lua` to run one or more models in real-time
off a webcam stream. To run this demo you need to use `qlua` instead of `th`:

```bash
qlua webcam_demo.lua -models models/instance_norm/candy.t7 -gpu 0
```

You can run multiple models at the same time by passing a comma-separated list
to the `-models` flag:

```bash
qlua webcam_demo.lua \
  -models models/instance_norm/candy.t7,models/instance_norm/udnie.t7 \
  -gpu 0
```

With a Pascal Titan X you can easily run four models in realtime at 640x480:

<div align='center'>
  <img src='images/webcam.gif' width='700px'>
</div>

The webcam demo depends on a few extra Lua packages:
- `camera`
- `qtlua`

You can install / update these packages by running:

```bash
luarocks install camera
luarocks install qtlua
```

The full set of options for this script is [described here](doc/flags.md#webcam_demolua).


## Training new models

You can [find instructions for training new models here](doc/training.md).

## Optimization-based Style Transfer

The script `slow_neural_style.lua` uses an optimization-based style-transfer method.

This script uses the same code for computing losses as the feedforward training
script, allowing for direct comparisons between feedforward style transfer networks
and optimization-based style transfer.

Improvements in this script:
- Removed external protobuf and `loadcaffe` dependencies
- Support for many CNN architectures, including ResNets

The full set of options for this script is [described here](doc/flags.md#slow_neural_stylelua).

## License

Free for personal or research use.


