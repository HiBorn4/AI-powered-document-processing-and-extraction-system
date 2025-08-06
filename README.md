# 🧠 i4Ideas: AI-Powered Document Processing & Extraction System

![Banner](/public/test1.png)

> An advanced AI-based system that extracts complex tabular and structured information from technical documents with unmatched precision. Built for manufacturing, steel, and QA/QC workflows.

---

### 📸 Visual Workflow — Document ➝ Table ➝ KPI Extraction

<table>
  <tr>
    <td><img src="/public/test2.png" width="100%"><br><center>📄 Raw Document</center></td>
  </tr>
  <tr>
    <td><img src="/public/table.png" width="100%"><br><center>✂️ Cropped Table</center></td>
  </tr>
  <tr>
    <td><img src="/public/final.png" width="100%"><br><center>🧠 KPIs</center></td>
  </tr>
</table>

---

## 🔍 Overview

**i4Ideas** is a state-of-the-art, production-grade document parsing system designed to extract meaningful data from industrial reports like tensile strength sheets, quality certificates, and lab reports. The system leverages deep learning, OCR, and vision-language models to output structured, human-verified JSON for downstream analytics or compliance.

This solution has been **field-tested across diverse document formats**, including JSW, Tata, Hyundai, and Jamshedpur technical data sheets.

---

## 🧠 Core Capabilities

- 📄 **PDF → Table/Image Extraction**
- 🔍 **Object Detection (YOLOv5 / ResNet50)** to locate tabular blocks
- 🧾 **PaddleOCR / Adobe OCR** integration for clean text extraction
- 🤖 **LLM-Powered Parsing (GPT)** using domain-specific prompts
- 📊 **Aesthetic JSON Output** with units and data validation
- 🔄 **Multi-format switch handling** (JSW, Tata, Hyundai, etc.)

---

## 📸 Supported Formats

| Company         | Keys Extracted                                                              |
|----------------|------------------------------------------------------------------------------|
| JSW Steel       | `Coil No`, `YS`, `UTS`, `EL`, `Ra`                                          |
| Tata Steel      | `Mother Coil`, `YS`, `UTS`, `EL`, `rBAR`, `n`, `Ra`                         |
| Jamshedpur      | `Coil No`, `YS`, `UTS`, `EL`, `rVALUE`, `NVALM`, `RaMICROM`                |
| Hyundai         | `Dimension`, `YP`, `TS`, `EL`, `Ra`                                         |
| Maharashtra     | `Product No`, `YP`, `TS`, `EL`, `Ra`                                        |

---

## 🚀 Installation

```bash
git clone https://github.com/HiBorn4/AI-powered-document-processing-and-extraction-system.git
cd AI-powered-document-processing-and-extraction-system
pip install -r requirements.txt
````

---

## 🎥 Demo

*(Add video preview here)*

---

## 🧪 Example Output

```json
{
  "NC65081000": {
    "YS": "347 MPa",
    "UTS": "294 MPa",
    "EL": "45%",
    "Ra": "67.39 μm"
  },
  "NC65082000": {
    "YS": "364 MPa",
    "UTS": "470 MPa",
    "EL": "80%",
    "Ra": "28.10 μm"
  }
}
```

---

## 📁 Project Structure

```
📦 AI-powered-document-processing-and-extraction-system
├── bulk_table_extraction.py         # Batch document processing
├── extract_keys.py                  # Prompt-based LLM key-value extraction
├── table_detection.py               # YOLO/ResNet table detection pipeline
├── pdf_to_image.py                  # Converts PDF to image for OCR
├── paddleocr_implemented/          # PaddleOCR model directory
├── jcap_tsr.py, tata_tsr.py, jsw_tsr.py  # Custom flows per company
├── image_postprocess.py             # Image cleaning and noise reduction
├── summarize_prompts.py             # Prompt engineering and format matching
```

---

## 📦 Tech Stack

* 🧠 **OpenAI GPT (Function Calling + Vision)**
* 🧾 **PaddleOCR / Adobe Acrobat OCR**
* 🖼️ **YOLOv5 / ResNet50** for layout detection
* 🧪 **Prompt Engineering** for reliable key mapping
* 📄 **PDF2Image**, **Pillow**, **Pandas**

---

## 🔐 Use Cases

* Steel Manufacturing QA Automation
* Compliance & Lab Test Report Digitization
* Quality Certificate Parsing & Validation
* Document Indexing for Enterprise Search

---

## 💼 Portfolio & Freelance Suitability

This project is ideal to showcase:

* LLMOps / Document AI skills
* OCR + Vision + GPT integration
* Custom multi-format prompt engineering
* PDF/Scan processing pipelines

---

## 📬 Contact

For collaboration or freelance work:

* 💼 **Upwork**: [HiBorn4](https://www.upwork.com/freelancers/~hiborn4)
* 💻 **GitHub**: [HiBorn4](https://github.com/HiBorn4)
* 📫 **Email**: [reach.hiborn@gmail.com](mailto:reach.hiborn4@gmail.com)

---

## 📝 License

MIT License. © 2025 HiBorn4