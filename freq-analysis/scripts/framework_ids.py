def generate_framework_id(existing_ids):
    for seq in range(1, 1000):
        fw_id = f"{seq:03d}"
        if fw_id not in existing_ids:
            existing_ids.add(fw_id)
            return fw_id

    raise ValueError("Maximum of 999 frameworks exceeded.")