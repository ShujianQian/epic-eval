//
// Created by Shujian Qian on 2024-03-31.
//

#include <cpu_auxiliary_range_index.h>

volatile mrcu_epoch_type active_epoch;
volatile mrcu_epoch_type globalepoch = 1;

kvtimestamp_t initial_timestamp;
kvepoch_t global_log_epoch;


namespace epic {

thread_local threadinfo *ti = nullptr;

}
