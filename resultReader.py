import sys
import json

if len(sys.argv) != 3 :
    print("Incorrect Argument!")
    print("Arguments: json_file transfer_type")
    exit()

transfer_type = sys.argv[2]
with open(sys.argv[1]) as file:
    line_content = [json.loads(lines) for lines in file.readlines()]
    total_duration = 0
    for line in line_content:
        for key, value in line.items():
            if transfer_type == "unified":
                if key == "Type" and value == 120:
                    start = float(line["CudaUvmGpuPageFaultEvent"]["startNs"])
                    end = float(line["CudaUvmGpuPageFaultEvent"]["endNs"])
                    total_duration += (end - start)
            elif transfer_type == "unified_pref":
                if key == "Type" and value == 80:
                    start = float(line["CudaEvent"]["startNs"])
                    end = float(line["CudaEvent"]["endNs"])
                    total_duration += (end - start)
            elif transfer_type == "direct":
                if key == "Type" and value == 80:
                    if line["CudaEvent"]["memcpy"]["srcKind"] == 0:
                        start = float(line["CudaEvent"]["startNs"])
                        end = float(line["CudaEvent"]["endNs"])
                        total_duration += (end - start)
            else:
                print("Invalid transfer type!", transfer_type)       
                print("Available transfer type: unified, unified_pref, direct")         
                exit()     
    print("Total Duration:", total_duration / 1000000000)