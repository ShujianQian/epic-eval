//
// Created by Shujian Qian on 2024-04-20.
//

#ifndef MICRO_CONFIG_H
#define MICRO_CONFIG_H

namespace epic::micro {

struct MicroConfig
{
  size_t num_records = 2'500'000;
  size_t num_txns = 100'000;
  size_t epochs = 5;
  double skew_factor = 0.0;
  uint32_t abort_percentage = 0;
};

}

#endif  // MICRO_CONFIG_H
