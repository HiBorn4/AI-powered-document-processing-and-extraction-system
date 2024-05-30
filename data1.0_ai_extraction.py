from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import os
import PyPDF2
import time
import os
import openai
import pandas as pd
import logging
import time
import json
import csv

# Measure the start time
overall_start_time = time.time()

def getchunkdata(parsed_data, temp_json_dict):
    c = 0
    table_list = []
    offset_list = []
    length_list = []
    for j in range(len((parsed_data.tables))):
        # print("enter getchunkdata j", j)
        table_list.append(j)
        offset = 0
        list_offset = []
        list_length = []

        for i in range(len(parsed_data.tables[j].spans)):
            list_offset.append(parsed_data.tables[j].spans[i].offset)
            list_length.append(parsed_data.tables[j].spans[i].length)
        offset = min(list_offset)
        length = sum(list_length)
        offset_list.append(offset)
        length_list.append(length)
        # rowCount = int(parsed_data.tables[j].row_count)
        # columnCount = int(parsed_data.tables[j].column_count)

    chunk_dict = {'chunks': []}
    offset_para = 0
    chunk_no = 0
    start_chunk = 0
    stop_chunk = 0

    string = ''
    for j in range(len((parsed_data.paragraphs))):
        # print("enter getchunkdata", parsed_data.paragraphs[j].spans[0].offset, parsed_data.paragraphs[j].spans[0].length)
        chunk_temp_dict = {}
        value2 = (parsed_data.paragraphs[j])
        offset1 = int(value2.spans[0].offset)
        length1 = int(value2.spans[0].length)
        # print ("offset:", offset1,length1, (offset1+length1))
        if (int(offset1) + int(length1) + 1) in offset_list:
            # print (offset1+length1+1)
            index_table = int(offset_list.index(int(offset1) + int(length1) + 1))
            start_chunk = offset_list[index_table]
            stop_chunk = int(offset_list[index_table]) + int(length_list[index_table]) + 1
            string += str(" ") + str(value2.content)
            chunk_temp_dict['chunk'] = string
            chunk_dict['chunks'].append(chunk_temp_dict)
            chunk_temp_dict = {}
            # print ("chunk me aagaya", chunk_dict['chunks'] )
            string = ''
            string = value2.content
            chunk_temp_dict['chunk'] = string
            table_name = f"table_{index_table}"
            # json1=globals().get(table_name)
            # print(table_name)
            chunk_temp_dict['table'] = table_name
            chunk_dict['chunks'].append(chunk_temp_dict)
            # print ("chunk me table", chunk_dict['chunks'] )
            string = ''
            offset_para = int(stop_chunk) - 1

        elif offset1 not in range(start_chunk, stop_chunk) or offset1 == 0:
            # print ("start,stop:", start_chunk, stop_chunk)
            if str(value2.content[int(len(value2.content)) - 1]) == ' ':
                string += str(value2.content)

            else:
                string += str(" ") + str(value2.content)
        # print("string", int(offset1 + length1) - offset_para)
        # print('chunk temp dict',chunk_temp_dict)
        if (int(offset1 + length1) - offset_para) < 1700:
            chunk_temp_dict['chunk'] = string
            chunk_dict['chunks'].append(chunk_temp_dict)
            chunk_no += 1
            string = str(value2.content)
            offset_para = int(offset1 + length1)
    return chunk_dict

def getrowforalltable(parsed_data):
    # print("enter getrowforalltable", len(parsed_data))
    offset_indicator = {'tables': []}
    c = 0
    temp_dict = {}

    for j in range(len(parsed_data)):
        temp_dict_table = {}
        temp_offset = {}
        temp_offset['table'] = int(j)
        temp_offset['span'] = str(parsed_data[j].spans)

        offset_indicator['tables'].append(temp_offset)

        rowCount = int(parsed_data[j].row_count)
        columnCount = int(parsed_data[j].column_count)
        # print(j,len(parsed_data['tables'][j]['cells']),rowCount,columnCount)
        first_row_counter = 0
        # print (j,rowCount, columnCount,parsed_data['tables'][j]['cells'][0]['content'])

        row = 0
        list_table_rows = []
        column_no = 0
        for i in range(int(rowCount * columnCount)):
            if i > (len(parsed_data[j].cells) - 1):
                break
            value1 = parsed_data[j].cells[i]
            # print (value1['content'], "i:",i, "row:", row,"rowIND:", value1['rowIndex'],"column:",column_no,"colInd:",value1['columnIndex'])
            if (value1.row_index) == row:
                column_index = int(value1.column_index)
                if column_no < column_index:
                    for i in range(column_index - column_no):
                        list_table_rows.append('')
                    list_table_rows.append(value1.content)
                    # print(value1['content'])
                    column_no = column_index + 1
                    # print (column_index)
                else:
                    # print ('else')
                    list_table_rows.append(value1.content)
                    column_no = column_index + 1
                    # print(list_table_rows)

            elif int(value1.row_index) > row:
                row = int(value1.row_index)
                if len(list_table_rows) < columnCount:
                    for i in range(columnCount - len(list_table_rows)):
                        list_table_rows.append('')

                column_no = 0
                column_index = int(value1.column_index)
                list_table_rows = []
                if column_no < column_index:
                    for i in range(column_index - column_no):
                        list_table_rows.append('')
                    list_table_rows.append(value1.content)
                    # print(value1['content'])
                    column_no = column_index + 1
                    # print (list_table_rows)
                else:
                    # print ('else')
                    list_table_rows.append(value1.content)
                    # print(value1['content'])
                    column_no = column_index + 1
                    # print(list_table_rows)
            if i > (len(parsed_data[j].cells) - 2):
                break

            if value1.row_index < 10:
                name1 = "Row 00" + str(value1.row_index)
            elif 100 > value1.row_index > 9:
                name1 = "Row 0" + str(value1.row_index)
            else:
                name1 = "Row " + str(value1.row_index)

            temp_dict_table[name1] = list_table_rows
            if int(value1.column_index) == int(columnCount - 1):
                first_row_counter = int(parsed_data[j].cells[i].row_index)
                # print (columnCount,int(parsed_data['tables'][j]['cells'][i]['columnIndex']))

            column_index = int(int(parsed_data[j].cells[i].column_index))
        tname = "table_" + str(j)
        temp_dict[tname] = temp_dict_table
    return temp_dict

def remove_rows_with_empty_cells(table_data):
    for table in table_data.values():
        for key in list(table):
            if len(table[key]) > 0:
                is_row_empty = True
                for cell in table[key]:
                    if cell:
                        is_row_empty = False
                        break
                if is_row_empty:
                    del table[key]
    return table_data

def remove_columns_with_empty_cells(table_data):
    for table in table_data.values():
        if len(table) > 0:
            i = 0
            j = len(table[list(table.keys())[0]])
            while i < j:
                is_column_empty = True
                for row in table.values():
                    if row[i]:
                        is_column_empty = False
                        break
                if is_column_empty:
                    for row in table.values():
                        row.pop(i)
                    i = i - 1
                    j = j - 1
                i = i + 1
    return table_data

def get_document_text(filename,formrecognizerservice,formrecognizerkey):
    offset = 0
    page_map = []
    # print("local filename :", filename)
    # print(f"Extracting text from '{filename}' using Azure Form Recognizer")
    form_recognizer_client = DocumentAnalysisClient(
        api_version="2022-08-31",
        endpoint=f"https://{formrecognizerservice}.cognitiveservices.azure.com/",
        credential=AzureKeyCredential(formrecognizerkey)
    )

    with open(filename, "rb") as f:
        poller = form_recognizer_client.begin_analyze_document("prebuilt-layout", document=f)

    form_recognizer_results = poller.result()
    # print("form_recognizer_results", form_recognizer_results)
    data = form_recognizer_results.to_dict()
    # print("data", data)
    table_data = getrowforalltable(form_recognizer_results.tables)
    if len(table_data) > 0:
        try:
            table_data = remove_rows_with_empty_cells(table_data)
            table_data = remove_columns_with_empty_cells(table_data)
        except Exception as e:
            # print(f"Error while removing empty rows/columns from table: {e}")
            pass
        return {"tables": table_data, "data": data}
    else:
        chunk = getchunkdata(form_recognizer_results, {})
        return ""
    
def split_pdf(input_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    with open(input_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader (file)

        # Iterate through each page
        for page_num in range(0, len(pdf_reader.pages), 2):
            # Create a new PDF writer object
            pdf_writer = PyPDF2.PdfWriter ()

            # Add the current page to the writer
            pdf_writer.add_page (pdf_reader.pages[page_num] )

            # Create the output PDF file name
            output_filename = os.path.join(output_folder, f'page_{page_num + 2}.pdf')

            # Write the new PDF to the output file
            with open(output_filename, 'wb') as output_file:
                pdf_writer.write(output_file)

    # print(f'PDF pages split and saved in {output_folder}')

def get_chat(Prompt, temp=0.7, tokens=2023):
    openai.api_type = "azure"
    openai.api_base = "https://cog-j7m2rbiz2kt7s.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = "cd30361e4ebb49c58f27d7193e978ee9"
    response = openai.ChatCompletion.create(
        engine="maichatgpt3516k",  # Use the appropriate model name
        messages=Prompt,
        temperature=temp,
        max_tokens=tokens,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response
    
def wholeExtraction(pdf_file_path):
    #Extraction of data using Azure Form Recognizier
    tic1=time.time()
    filepath=pdf_file_path
    f= get_document_text(filepath,"cog-fr-j7m2rbiz2kt7s","89df77c3061f49ea8668abca8977995f")
    # print(f)
    toc1=time.time()
    # print("toc1-tic1",toc1-tic1)
    print(f["tables"])
    print(f["data"]["content"])

    #Classication of Pages using GPT
    tic1=time.time()
    stack__message_json = []
    stack_Prompt = '''
    Please extract the following information from the attached pdf:
    Look for the Test Report No. and display it..
    Give a definitive and accurate answer and please double check the number before displaying the ouput..
    '''

    temp_dict = {
        "role": "system",
        "content": stack_Prompt
    }
    stack__message_json.append(temp_dict)
    temp_dict = {
        "role": "user",
        "content": str(f["data"]["content"])+str(f["tables"])
    }
    stack__message_json.append(temp_dict)
    # print("JSON: \n",type(text))
    summarize_stack = get_chat(stack__message_json, 0.1, 8000)["choices"][0]["message"]["content"]
    output_number = summarize_stack.split(':')[-1].strip()
    # print(output_number)
    summarize_stack=summarize_stack.replace(" ","")
    summarize_stack=summarize_stack[:5]+" "+summarize_stack[5:]
    print(summarize_stack)

    #Extraction of data using GPT
    summarize__message_json = []
    # 2. For differentiation you can differentiate Mother Coil Number and Coil Number by seeing the number of words, Mother Coil Number has 2 String whereas Coil Number has only one string. For Example, Mother Coil Number= [E955027 NC65330000, DC955027 NC46570000, .....] whereas Coil Number = [NC65330000, NC56389000, NC5896970000] 
    
    
    summarize_Prompt = '''
    Please extract the following information from the attached report strictly from the column labeled 'Result' and nowhere else:
    - Coil No.
    - YS (Mpa) (can also be referred to as YP)
    - UTS (Mpa) (can also be referred to as TS)
    - EL (%) 
    - rBAR (treated as RVALUE if "RBAR" is present)
    - NVALM (can be referred to as 'n')
    - Ra (μm) (can also be referred to as RaMICROM)

    Ensure that the extraction is accurate by cross-verifying the output from both the specific row and column where it is extracted before displaying the output.

    Important instructions:
    1. Do not confuse Heat No. or Mother Coil Number with Coil No. Extract ONLY Coil No. as entries for Coil No.
    - Example: 
        - Mother Coil Numbers: [E955027 NC65330000, DC955027 NC46570000, ...]
        - Coil Numbers: [NC65330000, NC56389000, NC5896970000]
    - Extract only Coil Number values.
    2. If a Coil No. is not immediately visible, search rigorously throughout the text to find it.
    3. Do not mix values from different rows. Each Coil No. should have its own unique set of YS (or YP), UTS (or TS), EL, rBAR, NVALM (or 'n'), and Ra (or RaMICROM) values from the same row.
    4. Verify that each attribute corresponds exactly to the correct Coil No. and is not mistakenly attributed to another Coil No.

    Generate an aesthetically pleasing JSON representation for a report, adhering to a standardized format. Ensure meticulous spacing and incorporate relevant units when necessary. Return the designated message "Data Unavailable" for any unavailable data in the table.

    The JSON format to be strictly followed:
    ```json
    {
        "(Coil No. Its Corresponding value)": {"YS (Mpa)": "value", "UTS (Mpa)": "value", "EL (%)": "value", "rBAR": "value", "NVALM": "value", "Ra (μm)": "value"},
        ...
    }
    ```

    Here is an example of the format:
    ```json
    {
        "NC65081000": {"YS (Mpa)": "347", "UTS (Mpa)": "294", "EL (%)": "45", "rBAR": "1.2", "NVALM": "0.95", "Ra (μm)": "67.39"},
        "NC65082000": {"YS (Mpa)": "364", "UTS (Mpa)": "470", "EL (%)": "80", "rBAR": "1.3", "NVALM": "0.96", "Ra (μm)": "28.10"}
    }
    '''

    # print("Prompt",summarize_Prompt)

    temp_dict = {
        "role": "system",
        "content": summarize_Prompt
    }
    summarize__message_json.append(temp_dict)
    temp_dict = {
        "role": "user",
        "content": str(f["data"]["content"])+str(f["tables"])
    }
    summarize__message_json.append(temp_dict)
    summarize_res = get_chat(summarize__message_json, 0.1, 8000)["choices"][0]["message"]["content"]
    print(summarize_res)
    return summarize_res

def save_json_to_file(json_string, file_name="extracted_data.json"):
    try:
        # Parse the JSON string into a dictionary
        json_data = json.loads(json_string)

        # Get the current directory
        current_directory = os.getcwd()

        # Define the full path for the file
        file_path = os.path.join(current_directory, file_name)

        # Save the JSON data to a file
        with open(file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"Data successfully saved to {file_path}")
    except json.JSONDecodeError as e:
        print(f"An error occurred while decoding JSON string: {e}")
    except Exception as e:
        print(f"An error occurred while saving data to JSON file: {e}")

def json_to_csv(json_file_path, csv_file_path):
    """
    Convert a JSON file into a CSV file where Coil No. is the row and its attributes are the columns.

    Args:
        json_file_path (str): Path to the input JSON file.
        csv_file_path (str): Path to the output CSV file.
    
    Returns:
        None
    """
    try:
        # Read the JSON data from the file
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # Extract the headers from the keys of the first dictionary
        headers = ["Coil No."] + list(next(iter(data.values())).keys())

        # Write the data to the CSV file
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            
            for coil_no, attributes in data.items():
                row = {"Coil No.": coil_no}
                row.update(attributes)
                writer.writerow(row)

        print(f"Data successfully saved to {csv_file_path}")

    except json.JSONDecodeError as e:
        print(f"An error occurred while decoding JSON file: {e}")
    except Exception as e:
        print(f"An error occurred while converting JSON to CSV: {e}")


# Main function to process the PDF files
def main():
    input_folder_path = 'data/data5.0'
    output_json_folder_path = 'output_json'
    output_csv_folder_path = 'output_csv'

    # Create output directories if they don't exist
    os.makedirs(output_json_folder_path, exist_ok=True)
    os.makedirs(output_csv_folder_path, exist_ok=True)

    # Get a list of all PDF files in the input folder
    pdf_files = [file for file in os.listdir(input_folder_path) if file.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        input_pdf_path = os.path.join(input_folder_path, pdf_file)
        results = []

        tic1 = time.time()

        # Extract and summarize the data from the PDF
        summarize_res = wholeExtraction(input_pdf_path)
        
        # Define file paths for saving JSON and CSV
        json_file_name = f"{os.path.splitext(pdf_file)[0]}.json"
        csv_file_name = f"{os.path.splitext(pdf_file)[0]}.csv"
        json_file_path = os.path.join(output_json_folder_path, json_file_name)
        csv_file_path = os.path.join(output_csv_folder_path, csv_file_name)

        # Save the summarized result to a JSON file
        save_json_to_file(summarize_res, json_file_path)
        
        # Clean up the summarized result for further processing
        scattered_text = summarize_res.replace("\n", " ")
        summarized_res_cleaned = summarize_res.replace("null", "None")
        summarized_res_cleaned = summarized_res_cleaned.replace("\n", "")
        summarized_res_cleaned = summarized_res_cleaned.replace(" ", "")
        
        # Convert the cleaned summarized result into a Python dictionary
        summarized_result = eval(summarized_res_cleaned)
        
        toc1 = time.time()
        print("Processing time for", pdf_file, ":", toc1 - tic1, "seconds")

        new_data = {}
        
        # Process each entry in the results list to construct a new data dictionary
        for entry in results:
            for value1 in entry:
                if value1 not in new_data:
                    new_data[value1] = []
                new_data[value1].append(entry[value1])

        newData = {}

        # Flatten the nested dictionaries into a single dictionary
        for outer_key, inner_list in new_data.items():
            outer_dict = {}
            for inner_dict in inner_list:
                inner_key = list(inner_dict.keys())[0]
                inner_value = list(inner_dict.values())[0] if list(inner_dict.values())[0] != 'DataUnavailable' else None
                outer_dict[inner_key] = inner_value
            newData[outer_key] = outer_dict

        # Create a DataFrame from the new data dictionary and transpose it
        df = pd.DataFrame(newData).transpose()

        # Convert JSON to CSV and save the CSV file
        json_to_csv(json_file_path, csv_file_path)

    # Measure the overall end time
    overall_end_time = time.time()

    # Calculate and print the overall execution time
    overall_execution_time = overall_end_time - overall_start_time
    print("Overall Execution Time:", overall_execution_time, "seconds")

# Run the main function
if __name__ == "__main__":
    overall_start_time = time.time()  # Measure the overall start time
    main()