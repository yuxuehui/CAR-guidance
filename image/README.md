# CAR Guidance — Image Experiment

## Controlling Rectified Flow on CelebA-HQ
The pre-trained generative model can be downloaded from [Rectified Flow CelebA-HQ](https://drive.google.com/file/d/1ryhuJGz75S35GEdWDLiq4XFrsbwPdHnF/view?usp=sharing) 
Just put it in ``` ./ ```

### Environment Setup

The project runs in a conda environment named `car_guidance_image` (Python 3.10, PyTorch
CUDA build). The provided `environment.yml` is a full export of the author's
environment; use it to reproduce the environment exactly:

```bash
conda env create -f environment.yml
conda activate car_guidance_image
```

> Note: if you hit `ImportError: ...libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`,
> it's a missing runtime `libittnotify.so` in some environments. The fix is to build a
> minimal stub library and preload it. For example:
>
> ```bash
> gcc -shared -fPIC -Wl,-soname,libittnotify.so -x c -o "${CONDA_PREFIX}/lib/libittnotify.so" - <<'EOF'
> int iJIT_NotifyEvent(int t, void *d) { (void)t; (void)d; return 0; }
> int iJIT_IsProfilingActive(void) { return 0; }
> unsigned int iJIT_GetNewMethodID(void) { static unsigned int id = 1; return id++; }
> EOF
> export LD_PRELOAD="${CONDA_PREFIX}/lib/libittnotify.so${LD_PRELOAD:+:${LD_PRELOAD}}"
> ```

### Run

You can use the demo image ``` ./demo/celeba.jpg ``` for running our model.

```
python main_data.py

```

### Dataset

The full Celeba-hq-1024 dataset can be downloaded from [kaggle celeba-hq dataset](https://www.kaggle.com/datasets/lamsimon/celebahq)
