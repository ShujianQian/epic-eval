import csv
import os
import re

benchmark = "tpcc"
database = "epic"
num_warehouses = 1
skew_factor = 0.0
fullread = "true"
# cpu_exec_num_threads = 16
cpu_exec_num_threads = 32
num_epochs = 5
num_txns = 100000
split_fields = "true"
commutative_ops = "false"
# num_records = 2500000
num_records = 10000000
exec_device = "gpu"

input_path = "./epic_epoch_size_output3"
output_path = "./epic_parsed_epoch_size_output3"
output_file_template = "output__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__r{}.txt"

INDEX_TSF_PATTERN = re.compile(r"Epoch 5 index_transfer time: (\d+) us")
INDEX_PATTERN = re.compile(r"Epoch 5 indexing time: (\d+) us")
INIT_TSF_PATTERN = re.compile(r"Epoch 5 init_transfer time: (\d+) us")
SUB_PATTEHR = re.compile(r"Epoch 5 submission time: (\d+) us")
INIT_PATTERN = re.compile(r"Epoch 5 initialization time: (\d+) us")
EXEC_TSF_PATTERN = re.compile(r"Epoch 5 exec_transfer time: (\d+) us")
EXEC_PATTERN = re.compile(r"Epoch 5 execution time: (\d+) us")


def parse_epic_output(lines):
    parsed_time = {
        "index_tsf": None,
        "index": None,
        "init_tsf": None,
        "sub": None,
        "init": None,
        "exec_tsf": None,
        "exec": None
    }
    for line in lines:
        match = INDEX_TSF_PATTERN.search(line)
        if match:
            parsed_time["index_tsf"] = int(match.group(1))
            continue

        match = INDEX_PATTERN.search(line)
        if match:
            parsed_time["index"] = int(match.group(1))
            continue

        match = INIT_TSF_PATTERN.search(line)
        if match:
            parsed_time["init_tsf"] = int(match.group(1))
            continue

        match = SUB_PATTEHR.search(line)
        if match:
            parsed_time["sub"] = int(match.group(1))
            continue

        match = INIT_PATTERN.search(line)
        if match:
            parsed_time["init"] = int(match.group(1))
            continue

        match = EXEC_TSF_PATTERN.search(line)
        if match:
            parsed_time["exec_tsf"] = int(match.group(1))
            continue

        match = EXEC_PATTERN.search(line)
        if match:
            parsed_time["exec"] = int(match.group(1))
            continue

    for key, value in parsed_time.items():
        if value is None:
            print("Error: key {} is None".format(key))
            return None
    return parsed_time


def epic_ycsb_experiment():
    database = "epic"

    for split_fields in ["true", "false"]:
        for benchmark in ["ycsba", "ycsbb", "ycsbc", "ycsbf"]:
            csv_rows = [
                ["alpha", "index_tsf", "index", "init_tsf", "sub", "init", "exec_tsf", "exec", "total_time", "num_txns",
                 "throughput"]]
            for skew_factor in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
                parsed_times = []
                for repeat in range(0, 3):
                    filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                           fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                           split_fields, commutative_ops, num_records,
                                                           exec_device, repeat)
                    filepath = os.path.join(input_path, filename)
                    with open(filepath, "r") as input_file:
                        lines = input_file.readlines()
                        parsed_time = parse_epic_output(lines)
                        if parsed_time is None:
                            print("Error: parsed_time is None for {}".format(filename))
                            exit(1)
                        parsed_times.append(parsed_time)
                avg_times = {}
                for key in parsed_times[0].keys():
                    avg_times[key] = round(
                        sum([parsed_time[key] for parsed_time in parsed_times]) / float(len(parsed_times)))
                print("avg_times: {}".format(avg_times))
                csv_row = [skew_factor, avg_times["index_tsf"], avg_times["index"], avg_times["init_tsf"],
                           avg_times["sub"], avg_times["init"], avg_times["exec_tsf"], avg_times["exec"]]
                total_time = sum([avg_times["index"], avg_times["init_tsf"], avg_times["sub"], avg_times["init"],
                                  avg_times["exec_tsf"], avg_times["exec"]])
                csv_row.append(total_time)
                csv_row.append(num_txns)
                csv_row.append(float(num_txns) / total_time)
                csv_rows.append(csv_row)
            csv_filename = "epic_{}_{}.csv".format(benchmark, "default" if split_fields == "false" else "split")
            csv_filepath = os.path.join(output_path, csv_filename)
            with open(csv_filepath, "w") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(csv_rows)


def epic_tpcc_experiment():
    database = "epic"
    benchmark = "tpcc"

    csv_rows = [["num_warehouses", "index_tsf", "index", "init_tsf", "sub", "init", "exec_tsf", "exec", "total_time",
                 "num_txns", "throughput"]]
    for num_warehouses in [1, 2, 4, 8, 16, 32, 64]:
        parsed_times = []
        for repeat in range(0, 3):
            filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                   fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                   split_fields, commutative_ops, num_records,
                                                   exec_device, repeat)
            filepath = os.path.join(input_path, filename)
            with open(filepath, "r") as input_file:
                lines = input_file.readlines()
                parsed_time = parse_epic_output(lines)
                if parsed_time is None:
                    print("Error: parsed_time is None for {}".format(filename))
                    exit(1)
                parsed_times.append(parsed_time)
        avg_times = {}
        for key in parsed_times[0].keys():
            avg_times[key] = round(sum([parsed_time[key] for parsed_time in parsed_times]) / float(len(parsed_times)))
        print("avg_times: {}".format(avg_times))
        csv_row = [num_warehouses, avg_times["index_tsf"], avg_times["index"], avg_times["init_tsf"],
                   avg_times["sub"], avg_times["init"], avg_times["exec_tsf"], avg_times["exec"]]
        total_time = sum(
            [avg_times["index"], avg_times["init_tsf"], avg_times["sub"], avg_times["init"], avg_times["exec_tsf"],
             avg_times["exec"]])
        csv_row.append(total_time)
        csv_row.append(num_txns)
        csv_row.append(float(num_txns) / total_time)
        csv_rows.append(csv_row)
    csv_filename = "epic_{}.csv".format(benchmark)
    csv_filepath = os.path.join(output_path, csv_filename)
    with open(csv_filepath, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_rows)


def gacco_ycsb_experiment():
    database = "gacco"
    num_txns = 32768

    for benchmark in ["ycsba", "ycsbb", "ycsbc", "ycsbf"]:
        csv_rows = [
            ["alpha", "index_tsf", "index", "init_tsf", "sub", "init", "exec_tsf", "exec", "total_time", "num_txns",
             "throughput"]]
        for skew_factor in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
            parsed_times = []
            for repeat in range(0, 3):
                filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                       fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                       split_fields, commutative_ops, num_records,
                                                       exec_device, repeat)
                filepath = os.path.join(input_path, filename)
                with open(filepath, "r") as input_file:
                    lines = input_file.readlines()
                    parsed_time = parse_epic_output(lines)
                    if parsed_time is None:
                        print("Error: parsed_time is None for {}".format(filename))
                        exit(1)
                    parsed_times.append(parsed_time)
            avg_times = {}
            for key in parsed_times[0].keys():
                avg_times[key] = round(
                    sum([parsed_time[key] for parsed_time in parsed_times]) / float(len(parsed_times)))
            print("avg_times: {}".format(avg_times))
            csv_row = [skew_factor, avg_times["index_tsf"], avg_times["index"], avg_times["init_tsf"],
                       avg_times["sub"], avg_times["init"], avg_times["exec_tsf"], avg_times["exec"]]
            total_time = sum(
                [avg_times["index"], avg_times["init_tsf"], avg_times["sub"], avg_times["init"], avg_times["exec_tsf"],
                 avg_times["exec"]])
            csv_row.append(total_time)
            csv_row.append(num_txns)
            csv_row.append(float(num_txns) / total_time)
            csv_rows.append(csv_row)
        csv_filename = "gacco_{}.csv".format(benchmark)
        csv_filepath = os.path.join(output_path, csv_filename)
        with open(csv_filepath, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_rows)


def gacco_tpcc_experiment():
    database = "gacco"
    num_txns = 32768

    for commutative_ops in ["true", "false"]:
        csv_rows = [
            ["num_warehouses", "n_index_tsf", "n_index", "n_init_tsf", "n_sub", "n_init", "n_exec_tsf", "n_exec",
             "n_total_time", "n_num_txns", "n_throughput", "p_index_tsf", "p_index", "p_init_tsf", "p_sub", "p_init",
             "p_exec_tsf", "p_exec",
             "p_total_time", "p_num_txns", "p_throughput", "total_time", "num_txns", "throughput"]]
        for num_warehouses in [1, 2, 4, 8, 16, 32, 64]:
            benchmark = "tpccn"
            parsed_times = []
            for repeat in range(0, 3):
                filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                       fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                       split_fields, commutative_ops, num_records,
                                                       exec_device, repeat)
                filepath = os.path.join(input_path, filename)
                with open(filepath, "r") as input_file:
                    lines = input_file.readlines()
                    parsed_time = parse_epic_output(lines)
                    if parsed_time is None:
                        print("Error: parsed_time is None for {}".format(filename))
                        exit(1)
                    parsed_times.append(parsed_time)
            avg_times = {}
            for key in parsed_times[0].keys():
                avg_times["n_" + key] = round(
                    sum([parsed_time[key] for parsed_time in parsed_times]) / float(len(parsed_times)))

            benchmark = "tpccp"
            parsed_times = []
            for repeat in range(0, 3):
                filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                       fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                       split_fields, commutative_ops, num_records,
                                                       exec_device, repeat)
                filepath = os.path.join(input_path, filename)
                with open(filepath, "r") as input_file:
                    lines = input_file.readlines()
                    parsed_time = parse_epic_output(lines)
                    if parsed_time is None:
                        print("Error: parsed_time is None for {}".format(filename))
                        exit(1)
                    parsed_times.append(parsed_time)
            for key in parsed_times[0].keys():
                avg_times["p_" + key] = round(
                    sum([parsed_time[key] for parsed_time in parsed_times]) / float(len(parsed_times)))
            if commutative_ops:
                n_total_time = sum(
                    [avg_times["n_index"], avg_times["n_init_tsf"], avg_times["n_exec_tsf"], avg_times["n_exec"]])
                p_total_time = sum(
                    [avg_times["p_index"], avg_times["p_init_tsf"], avg_times["p_exec_tsf"], avg_times["p_exec"]])
            else:
                n_total_time = sum(
                    [avg_times["n_index"], avg_times["n_init_tsf"], avg_times["n_sub"], avg_times["n_init"],
                     avg_times["n_exec_tsf"], avg_times["n_exec"]])
                p_total_time = sum(
                    [avg_times["p_index"], avg_times["p_init_tsf"], avg_times["p_sub"], avg_times["p_init"],
                     avg_times["p_exec_tsf"], avg_times["p_exec"]])
            n_throughput = float(num_txns) / n_total_time
            p_throughput = float(num_txns) / p_total_time
            total_time = n_total_time + p_total_time
            total_txns = num_txns * 2
            throughput = float(total_txns) / total_time

            csv_row = [num_warehouses, avg_times["n_index_tsf"], avg_times["n_index"], avg_times["n_init_tsf"],
                       avg_times["n_sub"], avg_times["n_init"], avg_times["n_exec_tsf"], avg_times["n_exec"],
                       n_total_time, num_txns, n_throughput,
                       avg_times["p_index_tsf"], avg_times["p_index"], avg_times["p_init_tsf"], avg_times["p_sub"],
                       avg_times["p_init"], avg_times["p_exec_tsf"], avg_times["p_exec"], p_total_time, num_txns,
                       p_throughput, total_time, total_txns, throughput]
            csv_rows.append(csv_row)
        csv_filename = "gacco_tpcc_{}.csv".format("commutative" if commutative_ops == "true" else "default")
        csv_filepath = os.path.join(output_path, csv_filename)
        with open(csv_filepath, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_rows)

def epic_cpu_tpcc_experiment():
    database = "epic"
    benchmark = "tpcc"
    exec_device = "cpu"

    csv_rows = [["num_warehouses", "index_tsf", "index", "init_tsf", "sub", "init", "exec_tsf", "exec", "total_time",
                 "num_txns", "throughput"]]
    for num_warehouses in [1, 2, 4, 8, 16, 32, 64]:
        parsed_times = []
        for repeat in range(0, 3):
            filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                   fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                   split_fields, commutative_ops, num_records,
                                                   exec_device, repeat)
            filepath = os.path.join(input_path, filename)
            with open(filepath, "r") as input_file:
                lines = input_file.readlines()
                parsed_time = parse_epic_output(lines)
                if parsed_time is None:
                    print("Error: parsed_time is None for {}".format(filename))
                    exit(1)
                parsed_times.append(parsed_time)
        avg_times = {}
        for key in parsed_times[0].keys():
            avg_times[key] = round(sum([parsed_time[key] for parsed_time in parsed_times]) / float(len(parsed_times)))
        print("avg_times: {}".format(avg_times))
        csv_row = [num_warehouses, avg_times["index_tsf"], avg_times["index"], avg_times["init_tsf"],
                   avg_times["sub"], avg_times["init"], avg_times["exec_tsf"], avg_times["exec"]]
        total_time = sum(
            [avg_times["index"], avg_times["init_tsf"], avg_times["sub"], avg_times["init"], avg_times["exec_tsf"],
             avg_times["exec"]])
        csv_row.append(total_time)
        csv_row.append(num_txns)
        csv_row.append(float(num_txns) / total_time)
        csv_rows.append(csv_row)
    csv_filename = "epic_cpu_{}.csv".format(benchmark)
    csv_filepath = os.path.join(output_path, csv_filename)
    with open(csv_filepath, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_rows)

def epic_cpu_ycsb_experiment():
    database = "epic"
    exec_device = "cpu"
    split_fields = "false"

    for benchmark in ["ycsba", "ycsbb", "ycsbc", "ycsbf"]:
        csv_rows = [
            ["alpha", "index_tsf", "index", "init_tsf", "sub", "init", "exec_tsf", "exec", "total_time", "num_txns",
             "throughput"]]
        for skew_factor in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
            parsed_times = []
            for repeat in range(0, 3):
                filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                       fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                       split_fields, commutative_ops, num_records,
                                                       exec_device, repeat)
                filepath = os.path.join(input_path, filename)
                with open(filepath, "r") as input_file:
                    lines = input_file.readlines()
                    parsed_time = parse_epic_output(lines)
                    if parsed_time is None:
                        print("Error: parsed_time is None for {}".format(filename))
                        exit(1)
                    parsed_times.append(parsed_time)
            avg_times = {}
            for key in parsed_times[0].keys():
                avg_times[key] = round(
                    sum([parsed_time[key] for parsed_time in parsed_times]) / float(len(parsed_times)))
            print("avg_times: {}".format(avg_times))
            csv_row = [skew_factor, avg_times["index_tsf"], avg_times["index"], avg_times["init_tsf"],
                       avg_times["sub"], avg_times["init"], avg_times["exec_tsf"], avg_times["exec"]]
            total_time = sum([avg_times["index"], avg_times["init_tsf"], avg_times["sub"], avg_times["init"],
                              avg_times["exec_tsf"], avg_times["exec"]])
            csv_row.append(total_time)
            csv_row.append(num_txns)
            csv_row.append(float(num_txns) / total_time)
            csv_rows.append(csv_row)
        csv_filename = "epic_cpu_{}.csv".format(benchmark)
        csv_filepath = os.path.join(output_path, csv_filename)
        with open(csv_filepath, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_rows)

def epic_ycsb_epoch_size_experiment():
    database = "epic"
    split_fields = "false"

    for benchmark in ["ycsbc", "ycsbf"]:
        for skew_factor in [0.0, 0.99]:
            csv_rows = [
                ["epoch_size", "index_tsf", "index", "init_tsf", "sub", "init", "exec_tsf", "exec", "total_time", "num_txns",
                 "throughput", "latency"]]
            for num_txns in [5000, 10000, 50000, 100000, 150000, 200000]:
                parsed_times = []
                for repeat in range(0, 3):
                    filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                           fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                           split_fields, commutative_ops, num_records,
                                                           exec_device, repeat)
                    filepath = os.path.join(input_path, filename)
                    with open(filepath, "r") as input_file:
                        lines = input_file.readlines()
                        parsed_time = parse_epic_output(lines)
                        if parsed_time is None:
                            print("Error: parsed_time is None for {}".format(filename))
                            exit(1)
                        parsed_times.append(parsed_time)
                avg_times = {}
                for key in parsed_times[0].keys():
                    avg_times[key] = round(
                        sum([parsed_time[key] for parsed_time in parsed_times]) / float(len(parsed_times)))
                print("avg_times: {}".format(avg_times))
                csv_row = [num_txns, avg_times["index_tsf"], avg_times["index"], avg_times["init_tsf"],
                           avg_times["sub"], avg_times["init"], avg_times["exec_tsf"], avg_times["exec"]]
                total_time = sum([avg_times["index"], avg_times["init_tsf"], avg_times["sub"], avg_times["init"],
                                  avg_times["exec_tsf"], avg_times["exec"]])
                latency = total_time / 1000.0 * 1.5
                csv_row.append(total_time)
                csv_row.append(num_txns)
                csv_row.append(float(num_txns) / total_time)
                csv_row.append(latency)
                csv_rows.append(csv_row)
            csv_filename = "epic_{}_{}_epoch_size.csv".format(benchmark, skew_factor)
            csv_filepath = os.path.join(output_path, csv_filename)
            with open(csv_filepath, "w") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(csv_rows)

def epic_tpcc_epoch_size_experiment():
    database = "epic"
    benchmark = "tpcc"

    for num_warehouses in [1, 64]:
        csv_rows = [
            ["epoch_size", "index_tsf", "index", "init_tsf", "sub", "init", "exec_tsf", "exec", "total_time", "num_txns",
             "throughput", "latency"]]
        for num_txns in [5000, 10000, 50000, 100000, 150000, 200000]:
            parsed_times = []
            for repeat in range(0, 3):
                filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                       fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                       split_fields, commutative_ops, num_records,
                                                       exec_device, repeat)
                filepath = os.path.join(input_path, filename)
                with open(filepath, "r") as input_file:
                    lines = input_file.readlines()
                    parsed_time = parse_epic_output(lines)
                    if parsed_time is None:
                        print("Error: parsed_time is None for {}".format(filename))
                        exit(1)
                    parsed_times.append(parsed_time)
            avg_times = {}
            for key in parsed_times[0].keys():
                avg_times[key] = round(
                    sum([parsed_time[key] for parsed_time in parsed_times]) / float(len(parsed_times)))
            print("avg_times: {}".format(avg_times))
            csv_row = [num_txns, avg_times["index_tsf"], avg_times["index"], avg_times["init_tsf"],
                       avg_times["sub"], avg_times["init"], avg_times["exec_tsf"], avg_times["exec"]]
            total_time = sum([avg_times["index"], avg_times["init_tsf"], avg_times["sub"], avg_times["init"],
                              avg_times["exec_tsf"], avg_times["exec"]])
            latency = total_time / 1000.0 * 1.5
            csv_row.append(total_time)
            csv_row.append(num_txns)
            csv_row.append(float(num_txns) / total_time)
            csv_row.append(latency)
            csv_rows.append(csv_row)
        csv_filename = "epic_{}_{}_epoch_size.csv".format(benchmark, num_warehouses)
        csv_filepath = os.path.join(output_path, csv_filename)
        with open(csv_filepath, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_rows)


if __name__ == "__main__":
    # Check if the directory exists
    if not os.path.exists(output_path):
        # If not, create the directory
        os.makedirs(output_path)
    # epic_ycsb_experiment()
    # epic_tpcc_experiment()
    # epic_cpu_tpcc_experiment()
    # epic_cpu_ycsb_experiment()
    # gacco_ycsb_experiment()
    # gacco_tpcc_experiment()
    epic_ycsb_epoch_size_experiment()
    epic_tpcc_epoch_size_experiment()
