

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