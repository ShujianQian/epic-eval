//
// Created by Shujian Qian on 2023-08-23.
//

#ifndef YCSB_TXN_H
#define YCSB_TXN_H

namespace epic::ycsb {

enum class YcsbOpType : uint8_t
{
    READ,                   /* read one field */
    FULL_READ,              /* read all fields */
    UPDATE,                 /* write one field */
    READ_MODIFY_WRITE,      /* read one field, modify that field, write that field */
    FULL_READ_MODIFY_WRITE, /* read all fields, modify one field, write that fields */
    INSERT                  /* insert a new record */
};

struct YcsbTxn
{
    uint32_t keys[10];
    uint8_t fields[10];
    YcsbOpType ops[10];
};

struct YcsbTxnParam
{
    uint32_t record_ids[10];
    uint8_t field_ids[10];
    YcsbOpType ops[10];
};

struct YcsbExecPlan
{
    union Plan
    {
        struct
        {
            uint32_t read_loc;
        } read_plan;
        struct
        {
            uint32_t read_locs[10];
        } full_read_plan;
        struct
        {
            uint32_t write_loc;
        } update_plan;
        struct
        {
            /* used for update when split_field is false */
            uint32_t read_loc;
            uint32_t write_loc;
        } copy_update_plan;
        struct
        {
            uint32_t read_loc;
            uint32_t write_loc;
        } read_modify_write_plan;
        struct
        {
            uint32_t read_locs[10];
            uint32_t write_loc;
        } full_read_modify_write_plan;
        struct
        {
            uint32_t write_loc;
        } insert_plan;
        struct
        {
            /* used for insert when split_field is true */
            uint32_t write_locs[10];
        } insert_full_plan;
    } plans[10];
};

static_assert(sizeof(YcsbExecPlan) == 440);

} // namespace epic::ycsb

#endif // YCSB_TXN_H
