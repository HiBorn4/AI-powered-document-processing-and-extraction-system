from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import PyPDF2
import time
import os
# import openai
import pandas as pd
import logging
import time
import json
import csv
import instructor
from pydantic import BaseModel
from openai import OpenAI


# Measure the start time
overall_start_time = time.time()

# # Define your desired output structure
# class UserInfo(BaseModel):
#     YS: int
#     UTS: int
#     EL: float
#     Ra: float

# def Instructor(extracted_text):
#     client = instructor.from_openai(OpenAI())

#     # Extract structured data from natural language
#     user_info = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         response_model=UserInfo,
#         messages=[{"role": "user", "content": extracted_text}],
#     )

#     print(user_info.YS)
#     print(user_info.UTS)