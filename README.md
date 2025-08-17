# AutoML Project

An automated machine learning (AutoML) framework built with Python that simplifies the process of data ingestion, analysis, and model building.

## Features

- **Data Ingestion**: Automated CSV file handling with session management
- **Data Analysis**: Comprehensive dataset analysis capabilities
- **Custom Logging**: Structured logging with JSON output and file/console handlers
- **Exception Handling**: Custom exception handling with detailed error tracking
- **Session Management**: Unique session IDs for data isolation and tracking

## Project Structure

```
automl/
├── app/
│   └── main.py                 # Main application entry point
├── src/
│   ├── datasetAnalysis/
│   │   ├── data_ingestion.py   # Dataset handling and CSV processing
│   │   └── data_analysis.py    # Data analysis functionality
│   └── dataCleaning/           # Data cleaning modules
├── logger/
│   └── customlogger.py         # Custom structured logging
├── expection/
│   └── customExpection.py      # Custom exception handling
├── data/                       # Data storage directory
├── logs/                       # Application logs
└── requirements.txt            # Project dependencies
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
- **Structlog**: Structured logging

## Usage

### Data Ingestion

```python
from src.datasetAnalysis.data_ingestion import datasetHandler

# Initialize handler
handler = datasetHandler()

# Load CSV file
df = handler.save_dataset(uploaded_file)
print(df.head())
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

## Configuration

### Environment Variables

- `DATA_STORAGE_PATH`: Custom data storage directory (default: `./data/datasetAnalysis`)

### Default Paths

- **Data Directory**: `data/datasetAnalysis/`
- **Logs Directory**: `logs/`
- **Session Format**: `session_id_YYYYMMDD_HHMMSS_<uuid>`

## Development

### Running Tests

```bash
# Run the data ingestion test
python src/datasetAnalysis/data_ingestion.py
```

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