# AutoML Project

An automated machine learning (AutoML) framework built with Python that simplifies the process of data ingestion, analysis, and model building.

## Features

- **Data Ingestion**: Automated CSV/Excel file handling with session management
- **AI-Powered Data Type Analysis**: LLM-based intelligent data type inference and conversion
- **Multi-LLM Support**: Integration with Google Gemini and Groq models
- **Custom Logging**: Structured logging with JSON output and file/console handlers
- **Exception Handling**: Custom exception handling with detailed error tracking
- **Session Management**: Unique session IDs for data isolation and tracking
- **Configuration Management**: YAML-based configuration system

## Project Structure

```
automl/
├── app/
│   └── main.py                     # Main application entry point
├── src/
│   ├── datasetAnalysis/
│   │   ├── data_ingestion.py       # Dataset handling and CSV/Excel processing
│   │   └── data_type_analysis.py   # AI-powered data type analysis
│   └── dataCleaning/               # Data cleaning modules
├── model/
│   └── models.py                   # Pydantic models for data validation
├── utils/
│   ├── model_loader.py             # LLM and embedding model loader
│   └── config_loader.py            # Configuration management
├── Propmt/
│   └── propmt_lib.py               # LLM prompt templates
├── config/
│   └── config.yml                  # Application configuration
├── logger/
│   └── customlogger.py             # Custom structured logging
├── expection/
│   └── customExpection.py          # Custom exception handling
├── data/                           # Data storage directory
├── logs/                           # Application logs
├── .env                            # Environment variables
└── requirements.txt                # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd automl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Dependencies

- **FastAPI & Uvicorn**: Web framework and ASGI server
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning library
- **XGBoost, LightGBM, CatBoost**: Advanced ML algorithms
- **PyCaret**: Low-code ML library
- **Matplotlib, Seaborn, Plotly**: Data visualization
- **LangChain**: LLM framework and integrations
- **Google Generative AI & Groq**: LLM providers
- **Pydantic**: Data validation and parsing
- **Structlog**: Structured logging
- **PyYAML**: Configuration file parsing

## Usage

### Data Ingestion

```python
from src.datasetAnalysis.data_ingestion import datasetHandler

# Initialize handler
handler = datasetHandler()

# Load CSV/Excel file
df = handler.save_dataset(uploaded_file)
print(df.head())
```

### AI-Powered Data Type Analysis

```python
from src.datasetAnalysis.data_type_analysis import DataTypeAnalyzer

# Initialize analyzer
analyzer = DataTypeAnalyzer("path/to/dataset.csv")

# Get AI recommendations for data types
recommendations = analyzer.analyze_data_type()

# Generate conversion code
code = analyzer.generate_conversion_code(recommendations)
print(code)

# Apply conversions
converted_df = analyzer.apply_conversions(df, recommendations)
```

### Custom Logging

```python
from logger.customlogger import CustomLogger

# Initialize logger
logger = CustomLogger().get_logger('MyModule')

# Log structured data
logger.info("Process started", user_id=123, filename="data.csv")
logger.error("Process failed", error="File not found")
```

### Exception Handling

```python
from expection.customExpection import AutoML_Exception

try:
    # Your code here
    pass
except Exception as e:
    raise AutoML_Exception("Custom error message", e)
```

## Key Components

### DatasetHandler Class

- **Session Management**: Creates unique session directories for data isolation
- **CSV Processing**: Validates and processes CSV files with UTF-8 encoding
- **Error Handling**: Comprehensive error handling with custom exceptions
- **Logging**: Structured logging for all operations

### Custom Logger

- **Structured Logging**: JSON-formatted logs with timestamps
- **Dual Output**: Logs to both file and console
- **Configurable**: Customizable log directory and formatting

### Custom Exception

- **Detailed Tracking**: Captures filename, line number, and full traceback
- **Integration**: Works seamlessly with the custom logger
- **Debugging**: Provides comprehensive error information

### AI Data Type Analyzer

- **LLM Integration**: Uses Google Gemini or Groq models for intelligent analysis
- **Smart Inference**: Analyzes sample data to recommend optimal data types
- **Code Generation**: Automatically generates pandas conversion code
- **Multi-format Support**: Handles CSV and Excel files
- **Validation**: Uses Pydantic models for structured output

## Configuration

### Environment Variables

- `DATA_STORAGE_PATH`: Custom data storage directory (default: `./data/datasetAnalysis`)
- `GROQ_API_KEY`: API key for Groq LLM services
- `GOOGLE_API_KEY`: API key for Google Generative AI
- `LLM_PROVIDER`: Choose LLM provider ("google" or "groq", default: "google")

### Configuration File

The `config/config.yml` file contains LLM settings:

```yaml
llm:
  google:
    provider: "google"
    model_name: "gemini-2.0-flash"
    temperature: 0.0
    max_output_tokens: 2048
  groq:
    provider: "groq"
    model_name: "deepseek-r1-distill-llama-70b"
    temperature: 0.0
    max_output_tokens: 2048
```

### Default Paths

- **Data Directory**: `data/datasetAnalysis/`
- **Logs Directory**: `logs/`
- **Session Format**: `session_id_YYYYMMDD_HHMMSS_<uuid>`

## Development

### Running Tests

```bash
# Run the data ingestion test
python src/datasetAnalysis/data_ingestion.py

# Run the data type analysis test
python src/datasetAnalysis/data_type_analysis.py

# Test model loader
python utils/model_loader.py
```

### Environment Setup

1. Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
LLM_PROVIDER=google
```

2. Configure your preferred LLM settings in `config/config.yml`

### Adding New Features

1. Create new modules in the appropriate `src/` subdirectory
2. Follow the existing logging and exception handling patterns
3. Update requirements.txt for new dependencies
4. Add tests and documentation

## Author

**Mayuresh Bairagi**

## Version

0.1.1

## License

This project is part of a college project for automated machine learning research and development.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## Support

For issues and questions, please check the logs directory for detailed error information and stack traces.