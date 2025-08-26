from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

change_data_type = ChatPromptTemplate.from_template("""
                                                    Based on the provided column names, datatypes, and sample values, determine which columns should be converted to a different datatype (e.g., string → number/date) and which should remain as string (e.g., names, categories).
                                                    Return ONLY valid JSON matching the exact schema below.
                                                    {format_instructions}  
                                                    Analyze this columns:
                                                    {Column_metadata}                                                
""")




PROMPT_REGISTRY = {
    'change_data_type': change_data_type
}