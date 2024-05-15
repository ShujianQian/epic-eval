import os
import sys
import subprocess

benchmark = "tpcc"
database = "epic"
num_warehouses = 1
skew_factor = 0.0
fullread = "true"
cpu_exec_num_threads = 32
num_epochs = 5
num_txns = 100000
split_fields = "true"
commutative_ops = "false"
num_records = 10000000
exec_device = "gpu"
num_repeat = 1

epic_driver_path = "./build/epic_driver"
epic_micro_driver_path = "./build/micro_driver"
output_path = "./epic_output"
cmd_template = "{} -b {} -d {} -w {} -a {} -r {} -c {} -e {} -s {} -f {} -m {} -n {} -x {}"
output_file_template = "output__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__r{}.txt"

micro_cmd_template = "{} -b {} -d {} -w {} -a {} -r {} -c {} -e {} -s {} -f {} -m {} -n {} -x {} -p {}"
micro_output_file_template = "output__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__p{}__r{}.txt"


def print_experiment_count():
    print_experiment_count.count += 1
    print("experiment count: ", print_experiment_count.count)


print_experiment_count.count = 0


def epic_ycsb_experiment():
    database = "epic"

    for split_fields in ["true", "false"]:
        for benchmark in ["ycsbf"]:
            for skew_factor in [0.99]:
                for repeat in range(0, num_repeat):
                    print_experiment_count()
                    cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                              fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                              commutative_ops, num_records, exec_device)
                    print(cmd)
                    command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                    print(command_output.stdout)
                    print(command_output.stderr, file=sys.stderr)
                    output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)


def epic_tpcc_experiment():
    database = "epic"
    benchmark = "tpcc"

    for num_warehouses in [64]:
        for repeat in range(0, num_repeat):
            print_experiment_count()
            cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                      fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                      commutative_ops, num_records, exec_device)
            print(cmd)
            command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            print(command_output.stdout)
            output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                          fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                          split_fields, commutative_ops, num_records,
                                                          exec_device, repeat)
            output_filepath = os.path.join(output_path, output_filename)
            with open(output_filepath, "w") as output_file:
                output_file.write(command_output.stdout)

def epic_tpcc_full_experiment():
    database = "epic"
    benchmark = "tpccfull"

    for num_warehouses in [64]:
        for repeat in range(0, num_repeat):
            print_experiment_count()
            cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                      fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                      commutative_ops, num_records, exec_device)
            print(cmd)
            command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            print(command_output.stdout)
            output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                          fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                          split_fields, commutative_ops, num_records,
                                                          exec_device, repeat)
            output_filepath = os.path.join(output_path, output_filename)
            with open(output_filepath, "w") as output_file:
                output_file.write(command_output.stdout)


def gacco_ycsb_experiment():
    database = "gacco"
    num_txns = 32768

    for benchmark in ["ycsbf"]:
        for skew_factor in [0.99]:
            for repeat in range(0, num_repeat):
                print_experiment_count()
                cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                          fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                          commutative_ops, num_records, exec_device)
                print(cmd)
                command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                print(command_output.stdout)
                output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                              fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                              split_fields, commutative_ops, num_records,
                                                              exec_device, repeat)
                output_filepath = os.path.join(output_path, output_filename)
                with open(output_filepath, "w") as output_file:
                    output_file.write(command_output.stdout)


def gacco_tpcc_experiment():
    database = "gacco"
    num_txns = 32768

    for benchmark in ["tpccn", "tpccp"]:
        for commutative_ops in ["true", "false"]:
            for num_warehouses in [64]:
                for repeat in range(0, num_repeat):
                    print_experiment_count()
                    cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                              fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                              commutative_ops, num_records, exec_device)
                    print(cmd)
                    command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                    print(command_output.stdout)
                    output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)


def epic_cpu_tpcc_experiment():
    database = "epic"
    benchmark = "tpcc"
    exec_device = "cpu"

    for num_warehouses in [64]:
        for repeat in range(0, num_repeat):
            print_experiment_count()
            cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                      fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                      commutative_ops, num_records, exec_device)
            print(cmd)
            command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            print(command_output.stdout)
            output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                          fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                          split_fields, commutative_ops, num_records,
                                                          exec_device, repeat)
            output_filepath = os.path.join(output_path, output_filename)
            with open(output_filepath, "w") as output_file:
                output_file.write(command_output.stdout)


def epic_cpu_tpcc_full_experiment():
    database = "epic"
    benchmark = "tpccfull"
    exec_device = "cpu"

    for num_warehouses in [64]:
        for repeat in range(0, num_repeat):
            print_experiment_count()
            cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                      fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                      commutative_ops, num_records, exec_device)
            print(cmd)
            command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            print(command_output.stdout)
            output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                          fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                          split_fields, commutative_ops, num_records,
                                                          exec_device, repeat)
            output_filepath = os.path.join(output_path, output_filename)
            with open(output_filepath, "w") as output_file:
                output_file.write(command_output.stdout)


def epic_cpu_ycsb_experiment():
    database = "epic"
    exec_device = "cpu"
    split_fields = "false"

    for benchmark in ["ycsbf"]:
        for skew_factor in [0.99]:
            for repeat in range(0, num_repeat):
                print_experiment_count()
                cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                          fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                          commutative_ops, num_records, exec_device)
                print(cmd)
                command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                print(command_output.stdout)
                output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                              fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                              split_fields, commutative_ops, num_records,
                                                              exec_device, repeat)
                output_filepath = os.path.join(output_path, output_filename)
                with open(output_filepath, "w") as output_file:
                    output_file.write(command_output.stdout)


def epic_ycsb_epoch_size_experiment():
    database = "epic"
    split_fields = "false"
    benchmark = "ycsbf"

    skew_factor_range = [0.99, 0.0]
    num_txns_ranges = [[70000],
                       [200000]]

    for skew_factor, num_txns_ranges in zip(skew_factor_range, num_txns_ranges):
        for num_txns in num_txns_ranges:
            for repeat in range(0, num_repeat):
                print_experiment_count()
                cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                          fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                          commutative_ops, num_records, exec_device)
                print(cmd)
                command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                print(command_output.stdout)
                output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                              fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                              split_fields, commutative_ops, num_records,
                                                              exec_device, repeat)
                output_filepath = os.path.join(output_path, output_filename)
                with open(output_filepath, "w") as output_file:
                    output_file.write(command_output.stdout)

def epic_tpcc_epoch_size_experiment():
    database = "epic"
    benchmark = "tpcc"

    num_warehouses_range = [1, 64]
    num_txns_ranges = [[40000],
                       [200000]]

    for num_warehouses, num_txns_range in zip(num_warehouses_range, num_txns_ranges):
        for num_txns in num_txns_range:
            for repeat in range(0, num_repeat):
                print_experiment_count()
                cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                          fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                          commutative_ops, num_records, exec_device)
                print(cmd)
                command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                print(command_output.stdout)
                output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                              fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                              split_fields, commutative_ops, num_records,
                                                              exec_device, repeat)
                output_filepath = os.path.join(output_path, output_filename)
                with open(output_filepath, "w") as output_file:
                    output_file.write(command_output.stdout)

def epic_microbenchmark():
    database = "epic"
    benchmark = "micro"
    skew_factor = 0.8
    for abort_rate in [50]:
        for repeat in range(0, num_repeat):
            cmd = micro_cmd_template.format(epic_micro_driver_path, benchmark, database, num_warehouses, skew_factor,
                                            fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                            commutative_ops, num_records, exec_device, abort_rate)
            print(cmd)
            command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            print(command_output.stdout)
            output_filename = micro_output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                split_fields, commutative_ops, num_records,
                                                                exec_device, abort_rate, repeat)
            output_filepath = os.path.join(output_path, output_filename)
            with open(output_filepath, "w") as output_file:
                output_file.write(command_output.stdout)
def gacco_tpcc_epoch_size_experiment():
    database = "gacco"

    num_warehouses_range = [1, 64]
    num_txns_ranges = [[4000],
                       [25000]]

    for benchmark in ["tpccn", "tpccp"]:
        for num_warehouses, num_txns_range in zip(num_warehouses_range, num_txns_ranges):
            for num_txns in num_txns_range:
                for repeat in range(0, num_repeat):
                    print_experiment_count()
                    cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                              fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                              commutative_ops, num_records, exec_device)
                    print(cmd)
                    command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                    print(command_output.stdout)
                    output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)

def gacco_ycsb_epoch_size_experiment():
    database = "gacco"
    split_fields = "false"
    benchmark = "ycsbf"

    skew_factor_range = [0.99, 0.0]
    num_txns_ranges = [[6000],
                       [45000]]

    for skew_factor, num_txns_ranges in zip(skew_factor_range, num_txns_ranges):
        for num_txns in num_txns_ranges:
            for repeat in range(0, num_repeat):
                print_experiment_count()
                cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                          fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                          commutative_ops, num_records, exec_device)
                print(cmd)
                command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                print(command_output.stdout)
                output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                              fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                              split_fields, commutative_ops, num_records,
                                                              exec_device, repeat)
                output_filepath = os.path.join(output_path, output_filename)
                with open(output_filepath, "w") as output_file:
                    output_file.write(command_output.stdout)


if __name__ == "__main__":
    # Check if the directory exists
    if not os.path.exists(output_path):
        # If not, create the directory
        os.makedirs(output_path)
    epic_tpcc_full_experiment()
    epic_cpu_tpcc_full_experiment()
    epic_tpcc_experiment()
    epic_cpu_tpcc_experiment()
    epic_ycsb_experiment()
    epic_cpu_ycsb_experiment()
    gacco_ycsb_experiment()
    gacco_tpcc_experiment()
    epic_ycsb_epoch_size_experiment()
    epic_tpcc_epoch_size_experiment()
    epic_microbenchmark()
    gacco_ycsb_epoch_size_experiment()
    gacco_tpcc_epoch_size_experiment()
