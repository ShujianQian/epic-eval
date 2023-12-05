//
// Created by Shujian Qian on 2023-12-05.
//

#include <benchmarks/ycsb_cpu_executor.h>

namespace epic::ycsb {

void CpuExecutor::execute(uint32_t epoch)
{
    uint32_t num_threads = config.cpu_exec_num_threads;
    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < num_threads; i++)
    {
        threads.emplace_back(&CpuExecutor::executionWorker, this, epoch, i);
    }

    for (auto &t : threads)
    {
        t.join();
    }
}

void CpuExecutor::executionWorker(uint32_t epoch, uint32_t thread_id)
{
    uint32_t num_txns = config.num_txns;
    uint32_t num_pieces = num_txns * 10;
    uint32_t num_threads = config.cpu_exec_num_threads;
    for (uint32_t piece_id = thread_id; piece_id < num_pieces; piece_id += num_threads)
    {
        uint32_t txn_id = piece_id / 10;
        uint32_t piece_offset = piece_id % 10;
        YcsbTxnParam *txn_param = reinterpret_cast<YcsbTxnParam *>(txn.getTxn(txn_id)->data);
        YcsbExecPlan *exec_plan = reinterpret_cast<YcsbExecPlan *>(plan.getTxn(txn_id)->data);
        YcsbValue value;
        switch (txn_param->ops[piece_offset])
        {
        case YcsbOpType::FULL_READ: {
            readFromTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].read_plan.read_loc, epoch, &value);
            break;
        }
        case YcsbOpType::FULL_READ_MODIFY_WRITE:
        case YcsbOpType::READ_MODIFY_WRITE: {
            readFromTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].read_modify_write_plan.read_loc,
                epoch, &value);
            memset(&value.data[txn_param->field_ids[piece_offset]], txn_id, sizeof(value.data[0]));
            writeToTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].read_modify_write_plan.write_loc,
                epoch, &value);
            break;
        }
        case YcsbOpType::UPDATE: {
            readFromTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].copy_update_plan.read_loc, epoch,
                &value);
            memset(&value.data[txn_param->field_ids[piece_offset]], txn_id, sizeof(value.data[0]));
            writeToTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].copy_update_plan.write_loc, epoch,
                &value);
            break;
        }
        default:
            throw std::runtime_error("epic::ycsb::CpuExecutor::executionWorker() found unknown op type.");
        }
        txn_param->record_ids[piece_offset] =
            value.data[txn_param->field_ids[piece_offset]][10]; /* prevent optimization */
    }
}

} // namespace epic::ycsb
