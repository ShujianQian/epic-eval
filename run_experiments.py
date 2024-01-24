import os
import subprocess

benchmark = "tpcc"
database = "epic"
num_warehouses = 1
skew_factor = 0.0
fullread = "true"
cpu_exec_num_threads = 16
num_epochs = 5
num_txns = 100000
split_fields = "true"
commutative_ops = "false"
num_records = 2500000
exec_device = "gpu"

epic_driver_path = "./build/epic_driver"
output_path = "./epic_output"
cmd_template = "{} -b {} -d {} -w {} -a {} -r {} -c {} -e {} -s {} -f {} -m {} -n {} -x {}"
output_file_template = "output__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__r{}.txt"


def print_experiment_count():
    print_experiment_count.count += 1
    print("experiment count: ", print_experiment_count.count)


print_experiment_count.count = 0


def epic_ycsb_experiment():
    database = "epic"

    for split_fields in ["true", "false"]:
        for benchmark in ["ycsba", "ycsbb", "ycsbc", "ycsbf"]:
            for skew_factor in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
                for repeat in range(0, 3):
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


def epic_tpcc_experiment():
    database = "epic"
    benchmark = "tpcc"

    for num_warehouses in [1, 2, 4, 8, 16, 32]:
        for repeat in range(0, 3):
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

    for benchmark in ["ycsba", "ycsbb", "ycsbc", "ycsbf"]:
        for skew_factor in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
            for repeat in range(0, 3):
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
            for num_warehouses in [1, 2, 4, 8, 16, 32]:
                for repeat in range(0, 3):
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

    for num_warehouses in [1, 2, 4, 8, 16, 32]:
        for repeat in range(0, 3):
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

    for benchmark in ["ycsba", "ycsbb", "ycsbc", "ycsbf"]:
        for skew_factor in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
            for repeat in range(0, 3):
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

    for num_txns in [5000, 10000, 50000, 100000, 150000, 200000, 250000]:
        for benchmark in ["ycsbc", "ycsbf"]:
            for skew_factor in [0.0, 0.99]:
                for repeat in range(0, 3):
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

    for num_txns in [5000, 10000, 50000, 100000, 150000, 200000, 250000]:
        for num_warehouses in [1, 64]:
            for repeat in range(0, 3):
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
    # epic_ycsb_experiment()
    # epic_tpcc_experiment()
    # epic_cpu_tpcc_experiment()
    # epic_cpu_ycsb_experiment()
    # gacco_ycsb_experiment()
    # gacco_tpcc_experiment()
    epic_ycsb_epoch_size_experiment()
    epic_tpcc_epoch_size_experiment()
