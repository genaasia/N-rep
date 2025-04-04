def parse_mac_schema(text, filter_dict, force_keep_all=False):
    lines = text.strip().split("\n")
    current_table = None

    new_schema = []

    for line in lines:
        # Check for table declaration
        if line.startswith("# Table:"):
            current_table = line.replace("# Table:", "").strip()
            if current_table in filter_dict:
                if new_schema:
                    new_schema.append("]")
                new_schema.append(line)
                new_schema.append("[")

        if current_table not in filter_dict:
            continue

        keep_all = filter_dict[current_table] == "keep_all" or force_keep_all
        # If we're inside a table definition and line contains a column definition
        if current_table is not None and line.strip().startswith("("):
            # Extract column name from the tuple format (column_name, description)
            try:
                column_name = (
                    line.strip().strip("(),").split(",")[0].split(":")[0].strip()
                )
                if keep_all:
                    new_schema.append(line)
                elif column_name in filter_dict[current_table]:
                    new_schema.append(line)
                # print(column_name)
            except IndexError:
                pass  # Skip malformed lines
    new_schema = new_schema + ["]"]

    return "\n".join(new_schema)


def parse_m_schema(text, filter_dict, force_keep_all=False):
    lines = text.strip().split("\n")
    current_table = None

    new_schema = []

    for line in lines:
        # Check for table declaration
        if line.startswith("# Table:"):
            current_table = line.replace("# Table:", "").strip()
            if current_table in filter_dict:
                if new_schema:
                    new_schema.append("]")
                new_schema.append(line)
                new_schema.append("[")

        if current_table not in filter_dict:
            continue

        keep_all = filter_dict[current_table] == "keep_all" or force_keep_all

        # If we're inside a table definition and line contains a column definition
        if current_table is not None and line.strip().startswith("("):
            # Extract column name from the tuple format (column_name, description)
            try:
                column_name = (
                    line.strip().strip("(),").split(",")[0].split(":")[0].strip()
                )
                if keep_all:
                    new_schema.append(line)
                elif column_name in filter_dict[current_table]:
                    new_schema.append(line)
                # print(column_name)
            except IndexError:
                pass  # Skip malformed lines
    new_schema = lines[:2] + new_schema + ["]"]

    new_fk = []
    if "【Foreign keys】" in text:
        fk_part = text.split("【Foreign keys】")[-1].strip().split("\n")
        for line in fk_part:
            try:
                left, right = line.split("=")
            except Exception as e:
                print(text)
                raise e
            left_table, left_col = left.split(".")
            right_table, right_col = right.split(".")

            if (
                left_table in filter_dict
                and (
                    filter_dict[left_table] == "keep_all"
                    or left_col in filter_dict[left_table]
                )
                and right_table in filter_dict
                and (
                    filter_dict[right_table] == "keep_all"
                    or right_col in filter_dict[right_table]
                )
            ):
                new_fk.append(line)

        if new_fk:
            new_schema.append("【Foreign keys】")
            new_schema += new_fk

    return "\n".join(new_schema)


# Example usage


def single_sample_test():
    # Sample text or you can read from a file
    sample_dict = {
        "schools": ["Magnet", "GSserved", "City", "School", "CDSCode"],
        "frpm": ["NSLP Provision Status", "School Name", "City", "CDSCode"],
    }

    ## TEST M SCHEMA
    print(f"FILTERED M SCHEMA:\n")
    with open("../data/dev_descriptions/m_schema/california_schools.txt", "r") as f:
        schema_text = f.read()
    new_schema = parse_m_schema(schema_text, sample_dict)
    print(new_schema)

    print(f"\n\n\n\n")

    ## TEST MAC SCHEMA
    print(f"FILTERED MAC SCHEMA:\n")
    with open("../data/dev_descriptions/mac_schema/california_schools.txt", "r") as f:
        schema_text = f.read()
    new_schema = parse_mac_schema(schema_text, sample_dict)
    print(new_schema)


def main():
    single_sample_test()

    # m_schema_inference_file = "../inferences/m_schema.json"
    # with open(m_schema_inference_file, 'r') as f:
    #     m_schema_inferences = f.read()

    # for inference in m_schema_inferences:
    #     db_id = inference["db_id"]
    #     # schema_text = inference["schema_text"]

    # mac_schema_inference_file = "../inferences/mac_schema.json"
    # with open(mac_schema_inference_file, 'r') as f:
    #     mac_schame_inferences = f.read()


if __name__ == "__main__":
    main()
