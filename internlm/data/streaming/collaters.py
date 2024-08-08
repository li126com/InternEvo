import torch


def pack_collate_fn(batch, micro_num, micro_bsz, seq_len):
    packed_length = micro_bsz * seq_len

    input_ids_list = []
    cu_seqlens_list = []
    indexes_list = []
    labels_list = []

    for b in batch:
        assert len(b["input_ids"]) == packed_length
        assert b["cu_seqlens"][0] == 0 and b["cu_seqlens"][-1] == packed_length
        assert len(b["indexes"]) == packed_length
        assert len(b["labels"]) == packed_length

        input_ids_list.append(torch.LongTensor(b["input_ids"]))
        cu_seqlens_list.append(torch.IntTensor(b["cu_seqlens"]))
        indexes_list.append(torch.IntTensor(b["indexes"]))
        labels_list.append(torch.LongTensor(b["labels"]))

    input_ids = torch.stack(input_ids_list)
    indexes = torch.stack(indexes_list)
    labels = torch.stack(labels_list)

    return {
        "input_ids": input_ids,
        "cu_seqlens": cu_seqlens_list,
        "indexes": indexes,
        "type_ids": torch.zeros(micro_num, micro_bsz * seq_len, dtype=torch.int64),
    }, labels
