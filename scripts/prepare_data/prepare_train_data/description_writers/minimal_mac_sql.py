import json
import os
import sqlite3
import sys
from copy import deepcopy
from typing import Dict, List, Tuple


def is_email(text: str) -> bool:
    """Check if text is an email address."""
    return "@" in text and "." in text


def is_valid_date_column(values: List[object]) -> bool:
    """Check if the column contains date values."""
    if not values:
        return False
    date_patterns = ["-", "/", "."]
    for value in values:
        if not isinstance(value, str):
            return False
        if not any(pattern in value for pattern in date_patterns):
            return False
    return True


class Selector:
    def __init__(
        self,
        data_path: str,
        tables_json_path: str,
        dataset_name: str,
        lazy: bool = False,
        without_selector: bool = False,
    ):
        self.data_path = data_path
        self.tables_json_path = tables_json_path
        self.dataset_name = dataset_name
        self.db2infos = {}
        self.db2dbjsons = {}
        self.init_db2jsons()
        if not lazy:
            self._load_all_db_info()
        self._message = {}
        self.without_selector = without_selector

    def init_db2jsons(self):
        if not os.path.exists(self.tables_json_path):
            print(f"{self.tables_json_path}")
            raise FileNotFoundError(f"tables.json not found in {self.tables_json_path}")
        with open(self.tables_json_path, "r") as f:
            data = json.load(f)
        for item in data:
            db_id = item["db_id"]
            table_names = item["table_names"]
            item["table_count"] = len(table_names)
            column_count_lst = [0] * len(table_names)
            for tb_idx, col in item["column_names"]:
                if tb_idx >= 0:
                    column_count_lst[tb_idx] += 1
            item["max_column_count"] = max(column_count_lst)
            item["total_column_count"] = sum(column_count_lst)
            item["avg_column_count"] = sum(column_count_lst) // len(table_names)
            self.db2dbjsons[db_id] = item

    def _get_column_attributes(self, cursor, table):
        cursor.execute(f"PRAGMA table_info(`{table}`)")
        columns = cursor.fetchall()
        columns_info = []
        primary_keys = []
        column_names = []
        column_types = []
        for column in columns:
            column_names.append(column[1])
            column_types.append(column[2])
            is_pk = bool(column[5])
            if is_pk:
                primary_keys.append(column[1])
            column_info = {
                "name": column[1],
                "type": column[2],
                "not_null": bool(column[3]),
                "primary_key": bool(column[5]),
            }
            columns_info.append(column_info)
        return column_names, column_types

    def _get_unique_column_values_str(
        self,
        cursor,
        table,
        column_names,
        column_types,
        json_column_names,
        is_key_column_lst,
    ):
        col_to_values_str_lst = []
        col_to_values_str_dict = {}
        key_col_list = [
            json_column_names[i] for i, flag in enumerate(is_key_column_lst) if flag
        ]
        len_column_names = len(column_names)

        for idx, column_name in enumerate(column_names):
            if column_name in key_col_list:
                continue

            lower_column_name: str = column_name.lower()
            if (
                lower_column_name.endswith("id")
                or lower_column_name.endswith("email")
                or lower_column_name.endswith("url")
            ):
                values_str = ""
                col_to_values_str_dict[column_name] = values_str
                continue

            sql = f"SELECT `{column_name}` FROM `{table}` GROUP BY `{column_name}` ORDER BY COUNT(*) DESC"
            cursor.execute(sql)
            values = cursor.fetchall()
            values = [value[0] for value in values]

            values_str = ""
            try:
                values_str = self._get_value_examples_str(values, column_types[idx])
            except Exception as e:
                print(f"\nerror: get_value_examples_str failed, Exception:\n{e}\n")

            col_to_values_str_dict[column_name] = values_str

        for k, column_name in enumerate(json_column_names):
            values_str = ""
            is_key = is_key_column_lst[k]

            if is_key:
                values_str = ""
            elif column_name in col_to_values_str_dict:
                values_str = col_to_values_str_dict[column_name]
            else:
                print(col_to_values_str_dict)
                print(
                    f"error: column_name: {column_name} not found in col_to_values_str_dict"
                )

            col_to_values_str_lst.append([column_name, values_str])

        return col_to_values_str_lst

    def _get_value_examples_str(self, values: List[object], col_type: str):
        if not values:
            return ""
        if len(values) > 10 and col_type in [
            "INTEGER",
            "REAL",
            "NUMERIC",
            "FLOAT",
            "INT",
        ]:
            return ""

        vals = []
        has_null = False
        for v in values:
            if v is None:
                has_null = True
            else:
                tmp_v = str(v).strip()
                if tmp_v == "":
                    continue
                else:
                    vals.append(v)
        if not vals:
            return ""

        if col_type in ["TEXT", "VARCHAR"]:
            new_values = []
            for v in vals:
                if not isinstance(v, str):
                    new_values.append(v)
                else:
                    if self.dataset_name == "spider":
                        v = v.strip()
                    if v == "":
                        continue
                    elif ("https://" in v) or ("http://" in v):
                        return ""
                    elif is_email(v):
                        return ""
                    else:
                        new_values.append(v)
            vals = new_values
            tmp_vals = [len(str(a)) for a in vals]
            if not tmp_vals:
                return ""
            max_len = max(tmp_vals)
            if max_len > 50:
                return ""

        if not vals:
            return ""

        vals = vals[:6]
        is_date_column = is_valid_date_column(vals)
        if is_date_column:
            vals = vals[:1]

        if has_null:
            vals.insert(0, None)

        val_str = str(vals)
        return val_str

    def _load_single_db_info(self, db_id: str) -> dict:
        table2coldescription = {}
        table2primary_keys = {}
        table_foreign_keys = {}
        table_unique_column_values = {}

        db_dict = self.db2dbjsons[db_id]
        important_key_id_lst = []
        keys = db_dict["primary_keys"] + db_dict["foreign_keys"]
        for col_id in keys:
            if isinstance(col_id, list):
                important_key_id_lst.extend(col_id)
            else:
                important_key_id_lst.append(col_id)

        db_path = f"{self.data_path}/{db_id}/{db_id}.sqlite"
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.cursor()

        table_names_original_lst = db_dict["table_names_original"]
        for tb_idx, tb_name in enumerate(table_names_original_lst):
            all_column_names_original_lst = db_dict["column_names_original"]
            all_column_names_full_lst = db_dict["column_names"]
            col2dec_lst = []

            pure_column_names_original_lst = []
            is_key_column_lst = []
            for col_idx, (root_tb_idx, orig_col_name) in enumerate(
                all_column_names_original_lst
            ):
                if root_tb_idx != tb_idx:
                    continue
                pure_column_names_original_lst.append(orig_col_name)
                if col_idx in important_key_id_lst:
                    is_key_column_lst.append(True)
                else:
                    is_key_column_lst.append(False)
                full_col_name: str = all_column_names_full_lst[col_idx][1]
                full_col_name = full_col_name.replace("_", " ")
                cur_desc_obj = [orig_col_name, full_col_name, ""]
                col2dec_lst.append(cur_desc_obj)
            table2coldescription[tb_name] = col2dec_lst

            table_foreign_keys[tb_name] = []
            table_unique_column_values[tb_name] = []
            table2primary_keys[tb_name] = []

            all_sqlite_column_names_lst, all_sqlite_column_types_lst = (
                self._get_column_attributes(cursor, tb_name)
            )
            col_to_values_str_lst = self._get_unique_column_values_str(
                cursor,
                tb_name,
                all_sqlite_column_names_lst,
                all_sqlite_column_types_lst,
                pure_column_names_original_lst,
                is_key_column_lst,
            )
            table_unique_column_values[tb_name] = col_to_values_str_lst

        foreign_keys_lst = db_dict["foreign_keys"]
        for from_col_idx, to_col_idx in foreign_keys_lst:
            from_col_name = all_column_names_original_lst[from_col_idx][1]
            from_tb_idx = all_column_names_original_lst[from_col_idx][0]
            from_tb_name = table_names_original_lst[from_tb_idx]

            to_col_name = all_column_names_original_lst[to_col_idx][1]
            to_tb_idx = all_column_names_original_lst[to_col_idx][0]
            to_tb_name = table_names_original_lst[to_tb_idx]

            table_foreign_keys[from_tb_name].append(
                (from_col_name, to_tb_name, to_col_name)
            )

        for pk_idx in db_dict["primary_keys"]:
            pk_idx_lst = []
            if isinstance(pk_idx, int):
                pk_idx_lst.append(pk_idx)
            elif isinstance(pk_idx, list):
                pk_idx_lst = pk_idx
            else:
                err_message = f"pk_idx: {pk_idx} is not int or list"
                print(err_message)
                raise Exception(err_message)
            for cur_pk_idx in pk_idx_lst:
                tb_idx = all_column_names_original_lst[cur_pk_idx][0]
                col_name = all_column_names_original_lst[cur_pk_idx][1]
                tb_name = table_names_original_lst[tb_idx]
                table2primary_keys[tb_name].append(col_name)

        cursor.close()

        result = {
            "desc_dict": table2coldescription,
            "value_dict": table_unique_column_values,
            "pk_dict": table2primary_keys,
            "fk_dict": table_foreign_keys,
        }
        return result

    def _load_all_db_info(self):
        print("\nLoading all database info...", file=sys.stdout, flush=True)
        db_ids = [
            "beer_factory",
            "books",
            "chicago_crime",
        ]
        for i in range(len(db_ids)):
            db_id = db_ids[i]
            db_info = self._load_single_db_info(db_id)
            self.db2infos[db_id] = db_info

    def _build_bird_table_schema_list_str(
        self, table_name, new_columns_desc, new_columns_val
    ):
        schema_desc_str = ""
        schema_desc_str += f"# Table: {table_name}\n"
        extracted_column_infos = []
        for (col_name, full_col_name, col_extra_desc), (_, col_values_str) in zip(
            new_columns_desc, new_columns_val
        ):
            col_extra_desc = (
                "And " + str(col_extra_desc)
                if col_extra_desc != "" and str(col_extra_desc) != "nan"
                else ""
            )
            col_extra_desc = col_extra_desc[:100]

            col_line_text = ""
            col_line_text += f"  ("
            col_line_text += f"{col_name},"

            if full_col_name != "":
                full_col_name = full_col_name.strip()
                col_line_text += f" {full_col_name}."
            if col_values_str != "":
                col_line_text += f" Value examples: {col_values_str}."
            if col_extra_desc != "":
                col_line_text += f" {col_extra_desc}"
            col_line_text += "),"
            extracted_column_infos.append(col_line_text)
        schema_desc_str += (
            "[\n" + "\n".join(extracted_column_infos).strip(",") + "\n]" + "\n"
        )
        return schema_desc_str

    def _get_db_desc_str(self, db_id: str) -> List[str]:
        if self.db2infos.get(db_id, {}) == {}:
            self.db2infos[db_id] = self._load_single_db_info(db_id)
        db_info = self.db2infos[db_id]
        desc_info = db_info["desc_dict"]
        value_info = db_info["value_dict"]
        pk_info = db_info["pk_dict"]
        fk_info = db_info["fk_dict"]
        tables_1, tables_2, tables_3 = (
            desc_info.keys(),
            value_info.keys(),
            fk_info.keys(),
        )
        assert set(tables_1) == set(tables_2)
        assert set(tables_2) == set(tables_3)

        schema_desc_str = ""
        db_fk_infos = []

        print(f"db_id: {db_id}")
        chosen_db_schem_dict = {}
        for (
            (table_name, columns_desc),
            (_, columns_val),
            (_, fk_info),
            (_, pk_info),
        ) in zip(
            desc_info.items(), value_info.items(), fk_info.items(), pk_info.items()
        ):
            all_columns = [name for name, _, _ in columns_desc]
            primary_key_columns = [name for name in pk_info]
            foreign_key_columns = [name for name, _, _ in fk_info]

            important_keys = primary_key_columns + foreign_key_columns

            new_columns_desc = []
            new_columns_val = []
            table_decision = "keep_all"
            # print(f"table_name: {table_name}")
            if table_decision == "drop_all":
                new_columns_desc = deepcopy(columns_desc[:6])
                new_columns_val = deepcopy(columns_val[:6])
            elif table_decision == "keep_all" or table_decision == "":
                new_columns_desc = deepcopy(columns_desc)
                new_columns_val = deepcopy(columns_val)
            else:
                llm_chosen_columns = table_decision
                # print(f"llm_chosen_columns: {llm_chosen_columns}")
                append_col_names = []
                for idx, col in enumerate(all_columns):
                    if col in important_keys:
                        new_columns_desc.append(columns_desc[idx])
                        new_columns_val.append(columns_val[idx])
                        append_col_names.append(col)
                    elif col in llm_chosen_columns:
                        new_columns_desc.append(columns_desc[idx])
                        new_columns_val.append(columns_val[idx])
                        append_col_names.append(col)
                    else:
                        pass

                if len(all_columns) > 6 and len(new_columns_val) < 6:
                    for idx, col in enumerate(all_columns):
                        if len(append_col_names) >= 6:
                            break
                        if col not in append_col_names:
                            new_columns_desc.append(columns_desc[idx])
                            new_columns_val.append(columns_val[idx])
                            append_col_names.append(col)

            chosen_db_schem_dict[table_name] = [
                col_name for col_name, _, _ in new_columns_desc
            ]

            schema_desc_str += self._build_bird_table_schema_list_str(
                table_name, new_columns_desc, new_columns_val
            )

            for col_name, to_table, to_col in fk_info:
                from_table = table_name
                if "`" not in str(col_name):
                    col_name = f"`{col_name}`"
                if "`" not in str(to_col):
                    to_col = f"`{to_col}`"
                fk_link_str = f"{from_table}.{col_name} = {to_table}.{to_col}"
                if fk_link_str not in db_fk_infos:
                    db_fk_infos.append(fk_link_str)
        fk_desc_str = "\n".join(db_fk_infos)
        schema_desc_str = schema_desc_str.strip()
        fk_desc_str = fk_desc_str.strip()

        return schema_desc_str, fk_desc_str, chosen_db_schem_dict
