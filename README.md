# Multi-gpus-pytorch
### 1. install [Apex](https://github.com/NVIDIA/apex)
```
>> git clone https://github.com/NVIDIA/apex
>> cd apex
```
#### For performance and full functionality, we recommend installing Apex with CUDA and C++ extensions via
```
>> pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
#### or APEX also supports a Python-only build via

```
>> pip install -v --disable-pip-version-check --no-cache-dir ./
```

### 2. train mnist using two gpus
```
>>./run.sh gpu1 gpu2
```
