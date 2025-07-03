import psutil
import cpuinfo
import pynvml


def print_cpu_info():
    """Prints cpu info"""
    info = cpuinfo.get_cpu_info()
    print("=== CPU INFO ===")
    print(f"CPU Name        : {info['brand_raw']}")
    print(f"Logical Cores   : {psutil.cpu_count(logical=True)}")
    print(f"Physical Cores  : {psutil.cpu_count(logical=False)}")
    print(f"Architecture    : {info['arch']}")
    print(f"Bits            : {info['bits']}")
    print()


def print_ram_info():
    """Prints ram info"""
    mem = psutil.virtual_memory()
    print("=== SYSTEM RAM ===")
    print(f"Total Memory    : {mem.total / 1e9:.2f} GB")
    print(f"Available       : {mem.available / 1e9:.2f} GB")
    print()


def print_gpu_info():
    """Prints gpu info"""
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print("=== GPU INFO (CUDA) ===")
        print(f"Detected GPUs   : {device_count}")
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"[GPU {i}] {name}")
            print(f"\tTotal Memory : {float(mem.total) / 1e9:.2f} GB")
            print(f"\tFree Memory  : {float(mem.free) / 1e9:.2f} GB")
            print(f"\tUsed Memory  : {float(mem.used) / 1e9:.2f} GB")
        print()
    except pynvml.NVMLError as e:
        print("Error accessing NVML:", str(e))


def main():
    print_cpu_info()
    print_ram_info()
    print_gpu_info()


if __name__ == "__main__":
    main()
