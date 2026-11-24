"""Admission uses independent DP rows, excluding padding when available."""


def route_upper_bound(local_rows, dp_size, topk, actual_rows=None):
    if actual_rows is None:
        rows = local_rows * dp_size
    else:
        rows = sum(int(n) for n in actual_rows)
        if len(actual_rows) != dp_size or any(int(n) < 0 or int(n) > local_rows for n in actual_rows):
            raise ValueError("Invalid unpadded DP token counts")
    return rows * topk
