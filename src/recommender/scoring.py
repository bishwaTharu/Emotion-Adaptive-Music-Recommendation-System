import torch
from ..data.va_binning import va_to_bin

def score_song(q_va, s_va, user, artist, ue_table, ea_table):
    v_bin, a_bin = va_to_bin(q_va[0], q_va[1])
    e = (v_bin, a_bin)

    mem_ue = ue_table.get(user, {}).get(e, 0.0)
    mem_ea = ea_table.get(artist, {}).get(e, 0.0)

    dist = torch.norm(q_va - s_va) ** 2
    return -dist + mem_ue + mem_ea
