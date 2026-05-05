<<<<<<< HEAD
"""
[BLK_M1_File.py]
Purpose: 
Author: Meng-Chi Ed Chen
Date: 
Reference:
    1.
    2.

Status: Working.
"""
import os, sys, json
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt



"""

To create a symbolic link (symlink) to Classes and Functions bank:
cmd /c mklink /J "00_Classes and Functions" "D:\01_Floor\a_Ed\09_EECS\10_Python\00_Classes and Functions"

"""


#[1] Direcotory and file - Start


def copy_file(path_src, path_dst):
    import shutil
    try:
        shutil.copy2(path_src, path_dst)
        #print(f'☑️ Copied file from {path_src} to {path_dst}')
    except Exception as e:
        print(f'⚠️ [Warning] Failed to copy file from {path_src} to {path_dst}. Error: {e}')



#[1] Direcotory and file - End





#[2] Text - Start

def save_dict_as_json(dict_obj, path_json, align_values=False):
    
    if align_values and isinstance(dict_obj, dict) and dict_obj:
        #[1] Calculate max key length for alignment
        cnt_spaces = 4
        max_key_len = max(len(key) for key in dict_obj.keys())
        
        lines = ["{"]
        items = list(dict_obj.items())
        for i, (key, value) in enumerate(items):
            # Convert string values to int if they look like integers
            if isinstance(value, str) and value.lstrip('-').isdigit():
                value = int(value)
            # Format the value using json.dumps for proper escaping
            value_str = json.dumps(value, ensure_ascii=False)
            # Calculate padding (1 extra space after colon)
            padding = max_key_len - len(key) + cnt_spaces
            # Add comma except for last item
            comma = "," if i < len(items) - 1 else ""
            lines.append(f'    "{key}":{" " * padding}{value_str}{comma}')
        lines.append("}")
        
        formatted_json = "\n".join(lines)
        with open(path_json, 'w', encoding='utf-8') as file:
            file.write(formatted_json)
    else:
        with open(path_json, 'w', encoding='utf-8') as file:
            json.dump(dict_obj, file, indent=4, ensure_ascii=False)
    
    return path_json

def read_json_as_dict(path_json):
    import json
    with open(path_json, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    if not content:
        print(f'⚠️ [Warning] JSON file is empty: {path_json}')
        return {}
    dict_obj = json.loads(content)
    return dict_obj


def format_dict_beautifully(dict_obj, keys_should_be_number=[],
                            value_char_limit=50,  
                            show_tabulate=True, cnt_spaces=4):
    """
    Generate two version of formated dictionary string
    1. tabulated simple table format for quick view in console.
    2. key-value pair format for better readability when saved as txt or json.
    Note:
    - list_keys_value: is a list of keys whose values should be treated as numbers (no quotes) in the key-value pair format. 
        By default, all values are treated as strings and enclosed in quotes.
    - value_char_limit: If a value is a string longer than this limit, 
        it will be truncated in the tabulated format for better display.
    """
    
    # [1] Tabulated table format for console display
    table_data = [[key, value] for key, value in dict_obj.items()]
    str_tabulated = tabulate(table_data, headers=["Key", "Value"], tablefmt="orgtbl")
    
    # [2] Key-value pair format for saving as txt/json
    max_key_length = max(len(str(key)) for key in dict_obj.keys())
    str_key_value = ""
    for key, value in dict_obj.items():
        i_spaces = max_key_length - len(str(key)) + cnt_spaces
        
        #[5] Treat as number (no quotes)
        if key in keys_should_be_number:
            value_str = str(value)
        #[6] Treat as string (enclose in quotes)
        elif isinstance(value, str):
            value_str = f"'{value}'"
        #[8] Treat None as 'None' string
        elif value is None:
            value_str = 'None'
        #[9] Other types (e.g., numbers, lists) are converted to string without quotes
        else:
            value_str = str(value)
        #[10] Truncate long string values in the key-value format for better readability
        if isinstance(value, str) and len(value) > value_char_limit:
            value_str = f"'{value[:value_char_limit]} ...'"
        str_key_value += f"'{key}':{' ' * i_spaces}{value_str}\n"
        
    #[3] Optionally print the tabulated format in console
    if show_tabulate:
        print(f'{str_tabulated}\n')
    
    return str_tabulated, str_key_value

#[2] Text - End





#[3] Array, Dataframe, Dictionary, and Excel - Start


def read_excel_sheet_into_df(path_xlsx, sheet_name=None, print_header=False):
    #[1] Read the Excel file into a DataFrame
    if sheet_name is None:
        sheet_name = 0  # Assuming 0 is the index of the first sheet
    df = pd.read_excel(path_xlsx, sheet_name=sheet_name)
    
    #[2] Tabulate the head of the DataFrame
    text_header = f"\n[read_excel_into_df] DataFrame Head:\n--\n"\
                  f"{tabulate(df.head(), headers='keys', tablefmt='orgtbl')}\n--\n"
    if print_header:
        print(text_header)
    del text_header
    return df


def save_df_as_excel_overwrite(df, path_xlsx, sheet_name, index=False):
    print('\n[save_df_as_excel_overwrite]')
    
    #[1] Make dir if not exist.
    dir_excel = os.path.dirname(path_xlsx)
    if not os.path.exists(dir_excel):
        os.makedirs(dir_excel)

    #[2] Try saving, wait if file is open
    while True:
        try:
            if os.path.exists(path_xlsx):
                with pd.ExcelWriter(path_xlsx, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=index)
            else:
                with pd.ExcelWriter(path_xlsx, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=index)
            break  # success: break loop

        except PermissionError:
            text_warning = f'\n⚠️[Warning] Cannot write to "{path_xlsx}"!\n'\
                            f'Did you close the excel fiel? Please close the file and hit Enter to try again.'
            print(text_warning)
            input('-- Press Enter after closing the Excel file...')

    print(f'-- Dataframe saved to {path_xlsx} in sheet {sheet_name}', end='\n\n')











def dict_covert_str_numbers_to_actual_numbers(input_dict, keys_should_be_number, path_dict_save):
    """
    keys_should_be_number are expect to be values (int, float, or bool).
    If they are not, raise a warning and ask user to confirm if they want to convert them to the correct type.
    """
    modified = False
    for key in keys_should_be_number:
        value = input_dict.get(key, None)
        if value is not None and isinstance(value, str):
            if value.isdigit():
                input_dict[key] = int(value)
                modified = True
            else:
                try:
                    input_dict[key] = float(value)
                    modified = True
                except ValueError:
                    print(f'⚠️ [Warning] The value for "{key}" is a string "{value}" but should be a number.')
                    continue
    if modified:
        save_dict_as_json(input_dict, path_dict_save, True)
        print(f'☑️ Updated dictionary saved to {path_dict_save}')
    return input_dict





def add_lookedup_values_to_dict_template(dict_template, dict_lookup):
    """
    dict_template: A template that might contain some missing values (None or "") that need to be filled.
    dict_lookup: values to fill into the template. The keys in dict_lookup should match the keys in dict_template that need to be filled.
    --
    A.Must raise error:
    1. A key in dict_lookup does not exist in dict_template. Need to update template or fix lookup.
    --
    B.Must raise warning:
    1. A key exists in both dictionaries but has different values. The value from dict_lookup will overwrite the one in dict_template.
    --
    C.No need to raise warning:
    1. A key exists in both dictionaries, but dict_template is empty or has the same value as dict_lookup for that key.
    """
    #[1] Make a copy to avoid mutating the original template
    result = dict_template.copy()
    for key, lookup_value in dict_lookup.items():
        if key not in dict_template:
            print(f'-- dict_template keys: {list(dict_template.keys())}')
            raise KeyError(f'❌ [Error] Key "{key}" in dict_lookup does not exist in dict_template. Update template or fix lookup.')
        template_value = dict_template[key]
        #[2] If template value is None or empty string, fill it
        if template_value in (None, ""):
            result[key] = lookup_value
        else:
            #[3] If values differ, warn and overwrite
            if template_value != lookup_value:
                print(f'⚠️ [Warning] Key "{key}" exists in both dictionaries with different values. Value from dict_lookup ("{lookup_value}") will overwrite dict_template ("{template_value}").')
                result[key] = lookup_value
            # If values are the same, do nothing
            # If template_value is not empty and same as lookup_value, no warning needed
    return result
    


def compare_dict_with_template(dict_to_check, dict_template, list_keys_allow_empty=[]):
    """
    Compare the keys of dict_to_check with dict_template and report any missing keys.
    --
    A.Must raise error:
    1. A key in dict_template is missing from dict_to_check.
    2. A key in dict_to_check does not exist in dict_template.
    --
    B.Must raise warning:
    1. A key in dict_to_check has an empty value (None or ""),
        but is not in the list of keys that are allowed to be empty.
    """
    # [A.1] Error: key in template missing from dict_to_check
    missing_keys = [key for key in dict_template.keys() if key not in dict_to_check]
    if missing_keys:
        raise KeyError(f'❌ [Error - Missing] The following keys from template are missing in dict_to_check: {missing_keys}')

    # [A.2] Error: key in dict_to_check not in template
    unexpected_keys = [key for key in dict_to_check.keys() if key not in dict_template]
    if unexpected_keys:
        raise KeyError(f'❌ [Error - Unexpected] The following keys in dict_to_check do not exist in dict_template: {unexpected_keys}')

    # [B.1] Warning: key in dict_to_check has empty value, not allowed
    cnt_key = len(dict_to_check)
    cnt_no_value = 0
    for key, value in dict_to_check.items():
        if value in (None, "") and key not in list_keys_allow_empty:
            cnt_no_value += 1
            print(f'⚠️ [Warning] Key "{key}" in dict_to_check has empty value (None or "") but is not allowed to be empty.')

    print(f'✅ All {cnt_key} keys are present. {cnt_no_value} keys have empty values (None or "").')




#[3] Array, Dataframe, Dictionary, and Excel - End





#[4] Interation - Start



def ask_for_input(question_message, readback=True, allow_empty=True):
    """
    Ed Chen's function, 2025-06-06.
    """
    print('\n[ask_for_input] ', end='')
    while True:
        str_find = input(question_message).strip()
        #[3] Check if str_find is not empty or if empty input is allowed
        if str_find or allow_empty:  
            if readback:
                print(f'-- User said: {str_find}')
            return str_find
        else:
            print('-- Please enter a non-empty value!')



def question_and_if_yes_action(yes_no_question, action_if_yes=None):
    """
    2025-0912-0958: Keep this function updated in F05_Interaction.py.
    """
    #[1] Asks a yes/no question, and if the answer is yes, executes the given action (function).
    answer = ask_for_input(yes_no_question)
    if any(keyword in answer.lower() for keyword in ['yes', 'ye', 'y', 'ok', 'okey', 'sure', 'go', 'aff', 'affirmative']):
        print('-- User agreed to proceed.', answer)
        return action_if_yes() if action_if_yes is not None else True
    else:
        sys.exit('-- User chose to exit.')




#[4] Interation - End





#[5] Performance and Resourcet - Start


#[5] Performance and Resourcet - End












=======
"""
[BLK_M1_File.py]
Purpose: 
Author: Meng-Chi Ed Chen
Date: 
Reference:
    1.
    2.

Status: Working.
"""
import os, sys, json
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt



"""

To create a symbolic link (symlink) to Classes and Functions bank:
cmd /c mklink /J "00_Classes and Functions" "D:\01_Floor\a_Ed\09_EECS\10_Python\00_Classes and Functions"

"""


#[1] Direcotory and file - Start


def copy_file(path_src, path_dst):
    import shutil
    try:
        shutil.copy2(path_src, path_dst)
        #print(f'☑️ Copied file from {path_src} to {path_dst}')
    except Exception as e:
        print(f'⚠️ [Warning] Failed to copy file from {path_src} to {path_dst}. Error: {e}')



#[1] Direcotory and file - End





#[2] Text - Start

def save_dict_as_json(dict_obj, path_json, align_values=False):
    
    if align_values and isinstance(dict_obj, dict) and dict_obj:
        #[1] Calculate max key length for alignment
        cnt_spaces = 4
        max_key_len = max(len(key) for key in dict_obj.keys())
        
        lines = ["{"]
        items = list(dict_obj.items())
        for i, (key, value) in enumerate(items):
            # Convert string values to int if they look like integers
            if isinstance(value, str) and value.lstrip('-').isdigit():
                value = int(value)
            # Format the value using json.dumps for proper escaping
            value_str = json.dumps(value, ensure_ascii=False)
            # Calculate padding (1 extra space after colon)
            padding = max_key_len - len(key) + cnt_spaces
            # Add comma except for last item
            comma = "," if i < len(items) - 1 else ""
            lines.append(f'    "{key}":{" " * padding}{value_str}{comma}')
        lines.append("}")
        
        formatted_json = "\n".join(lines)
        with open(path_json, 'w', encoding='utf-8') as file:
            file.write(formatted_json)
    else:
        with open(path_json, 'w', encoding='utf-8') as file:
            json.dump(dict_obj, file, indent=4, ensure_ascii=False)
    
    return path_json

def read_json_as_dict(path_json):
    import json
    with open(path_json, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    if not content:
        print(f'⚠️ [Warning] JSON file is empty: {path_json}')
        return {}
    dict_obj = json.loads(content)
    return dict_obj


def format_dict_beautifully(dict_obj, keys_should_be_number=[],
                            value_char_limit=50,  
                            show_tabulate=True, cnt_spaces=4):
    """
    Generate two version of formated dictionary string
    1. tabulated simple table format for quick view in console.
    2. key-value pair format for better readability when saved as txt or json.
    Note:
    - list_keys_value: is a list of keys whose values should be treated as numbers (no quotes) in the key-value pair format. 
        By default, all values are treated as strings and enclosed in quotes.
    - value_char_limit: If a value is a string longer than this limit, 
        it will be truncated in the tabulated format for better display.
    """
    
    # [1] Tabulated table format for console display
    table_data = [[key, value] for key, value in dict_obj.items()]
    str_tabulated = tabulate(table_data, headers=["Key", "Value"], tablefmt="orgtbl")
    
    # [2] Key-value pair format for saving as txt/json
    max_key_length = max(len(str(key)) for key in dict_obj.keys())
    str_key_value = ""
    for key, value in dict_obj.items():
        i_spaces = max_key_length - len(str(key)) + cnt_spaces
        
        #[5] Treat as number (no quotes)
        if key in keys_should_be_number:
            value_str = str(value)
        #[6] Treat as string (enclose in quotes)
        elif isinstance(value, str):
            value_str = f"'{value}'"
        #[8] Treat None as 'None' string
        elif value is None:
            value_str = 'None'
        #[9] Other types (e.g., numbers, lists) are converted to string without quotes
        else:
            value_str = str(value)
        #[10] Truncate long string values in the key-value format for better readability
        if isinstance(value, str) and len(value) > value_char_limit:
            value_str = f"'{value[:value_char_limit]} ...'"
        str_key_value += f"'{key}':{' ' * i_spaces}{value_str}\n"
        
    #[3] Optionally print the tabulated format in console
    if show_tabulate:
        print(f'{str_tabulated}\n')
    
    return str_tabulated, str_key_value

#[2] Text - End





#[3] Array, Dataframe, Dictionary, and Excel - Start


def read_excel_sheet_into_df(path_xlsx, sheet_name=None, print_header=False):
    #[1] Read the Excel file into a DataFrame
    if sheet_name is None:
        sheet_name = 0  # Assuming 0 is the index of the first sheet
    df = pd.read_excel(path_xlsx, sheet_name=sheet_name)
    
    #[2] Tabulate the head of the DataFrame
    text_header = f"\n[read_excel_into_df] DataFrame Head:\n--\n"\
                  f"{tabulate(df.head(), headers='keys', tablefmt='orgtbl')}\n--\n"
    if print_header:
        print(text_header)
    del text_header
    return df


def save_df_as_excel_overwrite(df, path_xlsx, sheet_name, index=False):
    print('\n[save_df_as_excel_overwrite]')
    
    #[1] Make dir if not exist.
    dir_excel = os.path.dirname(path_xlsx)
    if not os.path.exists(dir_excel):
        os.makedirs(dir_excel)

    #[2] Try saving, wait if file is open
    while True:
        try:
            if os.path.exists(path_xlsx):
                with pd.ExcelWriter(path_xlsx, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=index)
            else:
                with pd.ExcelWriter(path_xlsx, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=index)
            break  # success: break loop

        except PermissionError:
            text_warning = f'\n⚠️[Warning] Cannot write to "{path_xlsx}"!\n'\
                            f'Did you close the excel fiel? Please close the file and hit Enter to try again.'
            print(text_warning)
            input('-- Press Enter after closing the Excel file...')

    print(f'-- Dataframe saved to {path_xlsx} in sheet {sheet_name}', end='\n\n')











def dict_covert_str_numbers_to_actual_numbers(input_dict, keys_should_be_number, path_dict_save):
    """
    keys_should_be_number are expect to be values (int, float, or bool).
    If they are not, raise a warning and ask user to confirm if they want to convert them to the correct type.
    """
    modified = False
    for key in keys_should_be_number:
        value = input_dict.get(key, None)
        if value is not None and isinstance(value, str):
            if value.isdigit():
                input_dict[key] = int(value)
                modified = True
            else:
                try:
                    input_dict[key] = float(value)
                    modified = True
                except ValueError:
                    print(f'⚠️ [Warning] The value for "{key}" is a string "{value}" but should be a number.')
                    continue
    if modified:
        save_dict_as_json(input_dict, path_dict_save, True)
        print(f'☑️ Updated dictionary saved to {path_dict_save}')
    return input_dict





def add_lookedup_values_to_dict_template(dict_template, dict_lookup):
    """
    dict_template: A template that might contain some missing values (None or "") that need to be filled.
    dict_lookup: values to fill into the template. The keys in dict_lookup should match the keys in dict_template that need to be filled.
    --
    A.Must raise error:
    1. A key in dict_lookup does not exist in dict_template. Need to update template or fix lookup.
    --
    B.Must raise warning:
    1. A key exists in both dictionaries but has different values. The value from dict_lookup will overwrite the one in dict_template.
    --
    C.No need to raise warning:
    1. A key exists in both dictionaries, but dict_template is empty or has the same value as dict_lookup for that key.
    """
    #[1] Make a copy to avoid mutating the original template
    result = dict_template.copy()
    for key, lookup_value in dict_lookup.items():
        if key not in dict_template:
            print(f'-- dict_template keys: {list(dict_template.keys())}')
            raise KeyError(f'❌ [Error] Key "{key}" in dict_lookup does not exist in dict_template. Update template or fix lookup.')
        template_value = dict_template[key]
        #[2] If template value is None or empty string, fill it
        if template_value in (None, ""):
            result[key] = lookup_value
        else:
            #[3] If values differ, warn and overwrite
            if template_value != lookup_value:
                print(f'⚠️ [Warning] Key "{key}" exists in both dictionaries with different values. Value from dict_lookup ("{lookup_value}") will overwrite dict_template ("{template_value}").')
                result[key] = lookup_value
            # If values are the same, do nothing
            # If template_value is not empty and same as lookup_value, no warning needed
    return result
    


def compare_dict_with_template(dict_to_check, dict_template, list_keys_allow_empty=[]):
    """
    Compare the keys of dict_to_check with dict_template and report any missing keys.
    --
    A.Must raise error:
    1. A key in dict_template is missing from dict_to_check.
    2. A key in dict_to_check does not exist in dict_template.
    --
    B.Must raise warning:
    1. A key in dict_to_check has an empty value (None or ""),
        but is not in the list of keys that are allowed to be empty.
    """
    # [A.1] Error: key in template missing from dict_to_check
    missing_keys = [key for key in dict_template.keys() if key not in dict_to_check]
    if missing_keys:
        raise KeyError(f'❌ [Error - Missing] The following keys from template are missing in dict_to_check: {missing_keys}')

    # [A.2] Error: key in dict_to_check not in template
    unexpected_keys = [key for key in dict_to_check.keys() if key not in dict_template]
    if unexpected_keys:
        raise KeyError(f'❌ [Error - Unexpected] The following keys in dict_to_check do not exist in dict_template: {unexpected_keys}')

    # [B.1] Warning: key in dict_to_check has empty value, not allowed
    cnt_key = len(dict_to_check)
    cnt_no_value = 0
    for key, value in dict_to_check.items():
        if value in (None, "") and key not in list_keys_allow_empty:
            cnt_no_value += 1
            print(f'⚠️ [Warning] Key "{key}" in dict_to_check has empty value (None or "") but is not allowed to be empty.')

    print(f'✅ All {cnt_key} keys are present. {cnt_no_value} keys have empty values (None or "").')




#[3] Array, Dataframe, Dictionary, and Excel - End





#[4] Interation - Start



def ask_for_input(question_message, readback=True, allow_empty=True):
    """
    Ed Chen's function, 2025-06-06.
    """
    print('\n[ask_for_input] ', end='')
    while True:
        str_find = input(question_message).strip()
        #[3] Check if str_find is not empty or if empty input is allowed
        if str_find or allow_empty:  
            if readback:
                print(f'-- User said: {str_find}')
            return str_find
        else:
            print('-- Please enter a non-empty value!')



def question_and_if_yes_action(yes_no_question, action_if_yes=None):
    """
    2025-0912-0958: Keep this function updated in F05_Interaction.py.
    """
    #[1] Asks a yes/no question, and if the answer is yes, executes the given action (function).
    answer = ask_for_input(yes_no_question)
    if any(keyword in answer.lower() for keyword in ['yes', 'ye', 'y', 'ok', 'okey', 'sure', 'go', 'aff', 'affirmative']):
        print('-- User agreed to proceed.', answer)
        return action_if_yes() if action_if_yes is not None else True
    else:
        sys.exit('-- User chose to exit.')




#[4] Interation - End





#[5] Performance and Resourcet - Start


#[5] Performance and Resourcet - End












>>>>>>> 61910d571079a324c71ca9f2f536cc37805e49f4
