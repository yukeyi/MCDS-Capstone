import json
from pynvml import *

# following three functions are used for checking gpu usage
def getGpuUtilization(handle):
    try:
        util = nvmlDeviceGetUtilizationRates(handle)
        gpu_util = int(util.gpu)
    except NVMLError as err:
        error = handleError(err)
        gpu_util = error
    return gpu_util

def getMB(BSize):
    return BSize / (1024 * 1024)

def get_gpu_info(flag):
    print(flag)
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    data = []
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        gpu_util = getGpuUtilization(handle)
        one = {"gpuUtil": gpu_util}
        one["gpuId"] = i
        one["memTotal"] = getMB(meminfo.total)
        one["memUsed"] = getMB(meminfo.used)
        one["memFree"] = getMB(meminfo.total)
        one["temperature"] = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        data.append(one)
    data = {"gpuCount": deviceCount, "util": "Mb", "detail": data}
    print(json.dumps(data))