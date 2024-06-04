

### Data Analysis Summary

#### Data 1: JSW

**Attributes:**
- YS
- UTS
- EL
- Ra

**Note:** For EL, previous prompts considered GL values due to confusion. Using a new chat can improve accuracy.

summarize_Prompt = '''
Ensure that the extraction is accurate by following these steps:

1. Identify the Correct Column and Row:
   - Search for the exact column labeled 'Result'.
   - Extract information strictly from this column and verify its accuracy by cross-referencing the row data.

2. Match the Following Formats:
   - JSW Steel Coil:
     ```json
     {
         "(Coil No. Its Corresponding value)": {"YS": "value", "UTS": "value", "EL": "value", "Ra": "value"},
         ...
     }
     ```

Please extract the following information from the attached report strictly from the column labeled 'Result' and nowhere else:
- Coil No.
- YS (Mpa)
- UTS (Mpa) 
- EL (%)
- Ra (μm)

3. Important Instructions:
   - **Key Identification:**
     - Ensure only Coil No. and its corresponding values are extracted. Use the provided example as a guide.
   - **Accurate Value Extraction:**
     - Ensure each attribute (YS, UTS, EL, Ra) is extracted from the exact row corresponding to the Coil No.
     - Rigorously recheck the value of n. If there are any incorrect mappings, change them.
     - Recheck the values of EL and Ra for accuracy. Ensure the values are from the correct row.
   - **Avoid Adjacent Column Errors:**
     - Double-check that values are not mistakenly taken from adjacent columns.
   - **Verify Attribute Correspondence:**
     - Confirm that each attribute matches the correct Coil No. and avoid mixing values from different rows.

4. Generate an Aesthetically Pleasing JSON:
   - Format the JSON output for the report with meticulous spacing and relevant units.
   - For any unavailable data, return the message "Data Unavailable".

Example JSON Formats:

{
    "NCS6500037": {"YS": "440 Mpa", "UTS": "510 Mpa", "EL": "24.8%", "Ra": "1.6 μm"},
    ...
}


---

#### Data 2: Tata

**Attributes:**
- YS
- UTS
- EL
- rBAR
- Ra
- n

**Issues Noted:**
- Incorrect n values.
- Mixed-up attributes:
  - YS, UTS, n
  - n
  - n
  - n, UTS, YS

---

#### Data 3: Jamshedpur

**Attributes:**
- YS
- UTS
- EL
- RRVALUE
- RaMICRO
- NVALUE

---

#### Data 4: Posco

**Stage:** Second Stage

---

#### Data 5: Hyundai

**Attributes:**
- YP
- TS
- EL
- Ra

---

### Requirement Review: May 27th

**Goals:**
- One prompt
- "Data Unavailable" if an attribute is not present
- High accuracy in mapping
- Refine prompts

**Results:**
- All values were mapped incorrectly.

**Reason:**
1. Conflict in instructions:
   - Requirement for "Data Unavailable" if an attribute is not present.
   - Requirement to rigorously search for each attribute.
     - This led to mapping any found value because no attribute was marked as "Data Unavailable."
2. EL and GL cross-mapping due to combined prompt.
3. Failure to recognize Coil No. in some invoices, leading to dimension errors.

---

### Advantages

- High accuracy
- Robust performance

---

### New ML Model Implementation: May 28th

**Models:**
- ResNet50
- YOLOv8

**Steps:**

1. **Convert PDFs to Images**

2. **Tabular Data Detection Using YOLOv5 & ResNet50:**
   - Detect tabular data using YOLOv5, crop the image, and convert it to PDF.
   - Detect tabular data using ResNet50, crop the image, and convert it to PDF.
   - Compare YOLOv5 accuracy with ResNet50 accuracy.
   - Compare original image inputs with YOLOv5 results and ResNet50 results.

3. **Image Preprocessing**

4. **OCR Extraction Using Adobe Acrobat:**
   - Extract the Excel format of the cropped converted PDF.

5. **Python Code for Noise Removal:**
   - Remove noise from the extracted Excel file.
   - Maintain the general structure of the output.

---

### ResNet50 Model Details

**Labelling:**
- Classes

**Training:**
- 5 layers with 168 neurons each
- Freezing each layer
- Adjusting weights according to the output

---

### YOLOv8 Model Details

- High accuracy: Nearly 92%
- Requires a large dataset: Approximately 100 samples

---

### Considerations for May 29th

**Model Explanation:**
- Detailed explanation of the models used.
- Specific information regarding the documents.

**Switch Cases for Specific Company Formats:**
- Provide examples for each case.
  
---

This structured approach ensures clarity in identifying issues, applying models, and refining processes for better accuracy and robustness in data extraction and analysis.



EXTRACT TABLE FROM IMAGE DONE



--------------------------------------------------

summarize__message_json = []
    # 2. For differentiation you can differentiate Mother Coil Number and Coil Number by seeing the number of words, Mother Coil Number has 2 String whereas Coil Number has only one string. For Example, Mother Coil Number= [E955027 NC65330000, DC955027 NC46570000, .....] whereas Coil Number = [NC65330000, NC56389000, NC5896970000] 
    
    
    # summarize_Prompt = '''
    # Please extract the following information from the attached report strictly from the column labeled 'Result' and nowhere else:
    # - Coil No.
    # - YS (Mpa) (can also be referred to as YP)
    # - UTS (Mpa) (can also be referred to as TS)
    # - EL (%) 
    # - rBAR (treated as RVALUE if "RBAR" is present)
    # - NVALM (can be referred to as 'n')
    # - Ra (μm) (can also be referred to as RaMICROM)

    # Ensure that the extraction is accurate by cross-verifying the output from both the specific row and column where it is extracted before displaying the output.

    # Important instructions:
    # 1. Do not confuse Heat No. or Mother Coil Number with Coil No. Extract ONLY Coil No. as entries for Coil No.
    # - Example: 
    #     - Mother Coil Numbers: [E955027 NC65330000, DC955027 NC46570000, ...]
    #     - Coil Numbers: [NC65330000, NC56389000, NC5896970000]
    # - Extract only Coil Number values.
    # 2. If a Coil No. is not immediately visible, search rigorously throughout the text to find it.
    # 3. Do not mix values from different rows. Each Coil No. should have its own unique set of YS (or YP), UTS (or TS), EL, rBAR, NVALM (or 'n'), and Ra (or RaMICROM) values from the same row.
    # 4. Verify that each attribute corresponds exactly to the correct Coil No. and is not mistakenly attributed to another Coil No.

    # Generate an aesthetically pleasing JSON representation for a report, adhering to a standardized format. Ensure meticulous spacing and incorporate relevant units when necessary. Return the designated message "Data Unavailable" for any unavailable data in the table.

    # The JSON format to be strictly followed:
    # ```json
    # {
    #     "(Coil No. Its Corresponding value)": {"YS (Mpa)": "value", "UTS (Mpa)": "value", "EL (%)": "value", "rBAR": "value", "NVALM": "value", "Ra (μm)": "value"},
    #     ...
    # }
    # ```

    # Here is an example of the format:
    # ```json
    # {
    #     "NC65081000": {"YS (Mpa)": "347", "UTS (Mpa)": "294", "EL (%)": "45", "rBAR": "1.2", "NVALM": "0.95", "Ra (μm)": "67.39"},
    #     "NC65082000": {"YS (Mpa)": "364", "UTS (Mpa)": "470", "EL (%)": "80", "rBAR": "1.3", "NVALM": "0.96", "Ra (μm)": "28.10"}
    # }
    # '''

---------------------------------------------------------------

    # summarize_Prompt = '''
    # Please extract the following information from the attached report strictly from the table and nowhere else:
    # - YS (Yield Strength)
    # - UTS (Ultimate Tensile Strength)
    # - EL (Elongation)
    # - Ra (Roughness in micrometers)

    # Important Instructions:
    # 1. **Source**: Extract the data strictly from the table. Do not use any other part of the document.
    # 2. **Data Selection**:
    #     - **YS**: Extract the last entry under the YS (Yield Strength) column.
    #     - **UTS**: Extract the last entry under the UTS (Ultimate Tensile Strength) column.
    #     - **EL**: Extract the last entry under the EL (Elongation) column, which is represented as a percentage (%).
    #     - **Ra**: Extract the last entry under the Ra (Roughness) column, specifically measured in micrometers (μm).

    # 3. **Strict Column Selection**:
    #     - Ensure that the EL value is taken strictly from the EL column and not from any other column, including the last column of the table.
    #     - Similarly, ensure that the values for YS, UTS, and Ra are taken from their respective columns.

    # 4. **Accuracy**:
    #     - **Cross-Verification**: Ensure that the extracted values are accurate by cross-verifying the output from both the specific row and column where it is extracted.
    #     - **Consistency**: The values must correspond to the correct attributes (YS, UTS, EL, Ra) and should not be mixed up with any other values.

    # 5. **Format**:
    #     - **JSON Representation**: Generate an aesthetically pleasing JSON representation for the report. Adhere to a standardized format with meticulous spacing and relevant units.
    #     - **Unavailable Data**: If any of the data is unavailable in the table, return the message "Data Unavailable".

    # The JSON format to be strictly followed:
    # ```json
    # {
    #     "{"YS": "value", "UTS": "value", "EL": "value", "Ra": "value"},
    #     ...
    # }
    # ```

    # Here is an example of the format:
    # ```json
    # {
    #     {"YS": "347", "UTS": "294", "EL": "45", "Ra": "67.39"}
    # }
    # ```
    # '''

--------------------------------------------------

summarize_Prompt = '''
    Please extract the following information from the attached report strictly from the table and nowhere else:
    - YS (MPA)
    - UTS (MPA)
    - EL (%)
    - RBAR
    - NVALM
    - Ra MICROM


    Important Instructions:
    1. **Source**: Extract the data strictly from the table. Do not use any other part of the document.
    2. **Data Selection**:
        - **YS**: Extract the last entry under the YS (Yield Strength) column
        - **UTS**: Extract the last entry under the UTS (Ultimate Tensile Strength) column.
        - **EL**: Extract the last entry under the EL (Elongation) column, which is represented as a percentage (%).
        - **RBAR**: Extract the last entry under the RBAR Column.
        - **NVALM**: Extract the last entry under the NVALM Column.
        - **Ra MICROM**: Extract the last entry under the Ra (Roughness) column, specifically measured in micrometers (μm).

    3. **Strict Column Selection**:
        - Ensure that the EL value is taken strictly from the EL column and not from any other column, including the last column of the table.
        - Similarly, ensure that the values for YS, UTS, and Ra are taken from their respective columns.

    4. **Accuracy**:
        - **Cross-Verification**: Ensure that the extracted values are accurate by cross-verifying the output from both the specific row and column where it is extracted.
        - **Consistency**: The values must correspond to the correct attributes (YS, UTS, EL, rBAR, NVALMA, Ra) and should not be mixed up with any other values.

    5. **Format**:
        - **JSON Representation**: Generate an aesthetically pleasing JSON representation for the report. Adhere to a standardized format with meticulous spacing and relevant units.
        - **Unavailable Data**: If any of the data is unavailable in the table, return the message "Data Unavailable".

    The JSON format to be strictly followed:
    ```json
    {
        "{"YS": "value", "UTS": "value", "EL": "value", "rBAR": "value", "n": "value", "Ra": "value"},
        ...
    }
    ```

    Here is an example of the format:
    ```json
    {
        {"YS": "347", "UTS": "294", "EL": "45", "rBAR": "2.5", "NVALM": "0.32", "Ra": "67.39"}
    }
    ```
    '''