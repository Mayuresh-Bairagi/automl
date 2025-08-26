from langchain.prompts import ChatPromptTemplate

change_data_type = ChatPromptTemplate.from_template("""
You are a data type inference engine. 
Given metadata about DataFrame columns, analyze each column and decide if its datatype 
should be converted or left as-is. 
Output must be STRICTLY valid JSON matching the schema.:{return_instructions}

Guidelines:
1. Only choose among: object, integer, float, date, boolean. Never invent other dtypes.  
2. Be careful with tricky cases:
   - Numeric-looking strings (e.g., "125", "003") → integer unless leading zeros matter (then object).  
   - Decimals or scientific notation (e.g., "12.5", "1e-5") → float.  
   - Currency/percent (e.g., "$100", "75%") → float.  
   - Dates/times (e.g., "2020-01-01", "1 Jan 2020", "12/05/22", "2022-05-12 10:30") → date.  
   - Durations (e.g., "1h 20min", "5 days", "00:15:30") → date (time/duration).  
   - True/False, Yes/No, Y/N, 0/1 → boolean.  
   - Mixed numeric + text (e.g., "12kg", "5ft", "abc123") → object.  
   - IDs, phone numbers, zip codes, account numbers → object (even if numeric).  
   - Empty, null, or special placeholders ("NA", "null", "nan", "-") → ignore when inferring.  
   - Categories/names (e.g., "Apple", "Red", "Male") → object.  
3. Reason must clearly justify why the suggested dtype is chosen.  
4. Do not drop or modify sample values; only analyze them.  
5. Return ONLY valid JSON list of ColumnRecommendation objects. No extra text.  

Input metadata:
{Column_metadata}
""")






PROMPT_REGISTRY = {
    'change_data_type': change_data_type
}