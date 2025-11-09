# AutoML Project

An automated machine learning (AutoML) framework built with Python that simplifies the process of data ingestion, analysis, and model building.

## System Architecture

```mermaid
graph TB
    subgraph "Web Layer"
        A[FastAPI Web App] --> B[File Upload Endpoint]
    end
    
    subgraph "Data Processing Layer"
        C[Data Ingestion] --> D[Data Type Analysis]
        D --> E[Feature Engineering]
        E --> F[Target Variable Detection]
        F --> G[Feature Selection]
        G --> H[ML Classification]
    end
    
    subgraph "AI/LLM Layer"
        I[Google Gemini]
        J[Groq Models]
        K[LangChain Framework]
    end
    
    subgraph "Storage Layer"
        L[Session Data]
        M[Processed Files]
        N[Trained Models]
        O[Logs]
    end
    
    subgraph "Analysis Layer"
        P[EDA Reports]
        Q[Model Performance]
        R[Feature Rankings]
    end
    
    B --> C
    D -.-> I
    D -.-> J
    E -.-> K
    F -.-> I
    G -.-> J
    C --> L
    E --> M
    H --> N
    A --> O
    E --> P
    H --> Q
    G --> R
```

## Workflow Diagram

```mermaid
flowchart TD
    Start([Upload CSV/Excel File]) --> A[Data Ingestion]
    A --> B[Session Creation]
    B --> C[AI Data Type Analysis]
    C --> D[Data Type Conversion]
    D --> E[Feature Engineering]
    E --> F{Datetime Columns?}
    F -->|Yes| G[Extract Date Features]
    F -->|No| H{Object Columns?}
    G --> H
    H -->|Yes| I[LLM Feature Extraction]
    H -->|No| J[Target Variable Detection]
    I --> J
    J --> K[Problem Type Classification]
    K --> L[Feature Selection]
    L --> M[Statistical Analysis]
    M --> N[LLM Feature Ranking]
    N --> O[Data Preprocessing]
    O --> P[Label Encoding]
    P --> Q[MinMax Scaling]
    Q --> R[Model Training]
    R --> S[Hyperparameter Tuning]
    S --> T[Model Evaluation]
    T --> U[Model Persistence]
    U --> V[EDA Report Generation]
    V --> End([Complete AutoML Pipeline])
    
    style Start fill:#e1f5fe
    style End fill:#c8e6c9
    style C fill:#fff3e0
    style I fill:#fff3e0
    style J fill:#fff3e0
    style N fill:#fff3e0
```

## UML Diagrams

### Class Diagram

```mermaid
classDiagram
    class FastAPIApp {
        +app: FastAPI
        +upload_file(file: UploadFile)
        +get_status(session_id: str)
        +start_server()
    }
    
    class DatasetHandler {
        -data_storage_path: str
        -logger: Logger
        +save_dataset(file: UploadFile) DataFrame
        +create_session_directory() str
        +validate_file(file: UploadFile) bool
        +read_csv(file_path: str) DataFrame
        +read_excel(file_path: str) DataFrame
    }
    
    class DataTypeAnalyzer {
        -dataset_path: str
        -llm_model: LLMModel
        -logger: Logger
        +analyze_data_type() DataTypeRecommendation
        +generate_conversion_code(recommendations: dict) str
        +apply_conversions(df: DataFrame, recommendations: dict) DataFrame
        -get_sample_data(df: DataFrame) DataFrame
        -validate_recommendations(recommendations: dict) bool
    }
    
    class FeatureEngineer1 {
        -dataset_path: str
        -dataframe: DataFrame
        -session_id: str
        -llm_model: LLMModel
        +generate_features() tuple[DataFrame, str]
        +process_datetime_columns(df: DataFrame) DataFrame
        +process_object_columns(df: DataFrame) DataFrame
        +extract_weight_values(df: DataFrame) DataFrame
        +extract_duration_values(df: DataFrame) DataFrame
        -create_session_id() str
        -save_processed_data(df: DataFrame) str
    }
    
    class TargetVariable {
        -session_id: str
        -llm_model: LLMModel
        -logger: Logger
        +get_target_variable(problem_statement: str) tuple[dict, DataFrame]
        +classify_problem_type(target_col: str, df: DataFrame) str
        -analyze_column_characteristics(df: DataFrame) dict
        -validate_target_selection(target: str, df: DataFrame) bool
    }
    
    class FeatureSelector {
        -session_id: str
        -problem_statement: str
        -llm_model: LLMModel
        +llm_response() dict
        +statistical_feature_selection(df: DataFrame, target: str) list
        +correlation_analysis(df: DataFrame) dict
        +mutual_info_selection(X: DataFrame, y: Series) list
        +chi2_selection(X: DataFrame, y: Series) list
        -combine_selection_methods(results: list) list
    }
    
    class AutoMLClassifier {
        -session_id: str
        -problem_statement: str
        -target_result: dict
        -dataframe: DataFrame
        -models: dict
        +train_models() tuple[DataFrame, dict, dict]
        +preprocess_data(df: DataFrame) tuple[DataFrame, dict]
        +train_single_model(model_name: str, X: DataFrame, y: Series) dict
        +evaluate_model(model: object, X_test: DataFrame, y_test: Series) dict
        +save_model(model: object, model_name: str) str
        -setup_models() dict
        -hyperparameter_tuning(model: object, X: DataFrame, y: Series) object
    }
    
    class EDA {
        -session_id: str
        -logger: Logger
        +generate_report() str
        +create_profile_report(df: DataFrame) ProfileReport
        +save_html_report(report: ProfileReport) str
        -load_processed_data() DataFrame
        -validate_data_quality(df: DataFrame) dict
    }
    
    class LLMModel {
        <<interface>>
        +generate_response(prompt: str) str
        +validate_response(response: str) bool
    }
    
    class GoogleGeminiModel {
        -api_key: str
        -model_name: str
        -temperature: float
        +generate_response(prompt: str) str
        +configure_model(config: dict) void
    }
    
    class GroqModel {
        -api_key: str
        -model_name: str
        -temperature: float
        +generate_response(prompt: str) str
        +configure_model(config: dict) void
    }
    
    class CustomLogger {
        -log_directory: str
        -log_level: str
        +get_logger(module_name: str) Logger
        +configure_handlers() list
        +format_log_message(message: str, **kwargs) str
    }
    
    class AutoML_Exception {
        -error_message: str
        -original_exception: Exception
        -filename: str
        -line_number: int
        +__init__(message: str, exception: Exception)
        +get_error_details() dict
    }
    
    class ConfigLoader {
        +load_config(config_path: str) dict
        +get_llm_config(provider: str) dict
        +validate_config(config: dict) bool
    }
    
    class SessionManager {
        -base_path: str
        +create_session() str
        +get_session_path(session_id: str) str
        +cleanup_old_sessions() void
        +validate_session(session_id: str) bool
    }
    
    %% Relationships
    FastAPIApp --> DatasetHandler : uses
    DatasetHandler --> DataTypeAnalyzer : creates
    DataTypeAnalyzer --> FeatureEngineer1 : triggers
    FeatureEngineer1 --> TargetVariable : flows to
    TargetVariable --> FeatureSelector : flows to
    FeatureSelector --> AutoMLClassifier : flows to
    AutoMLClassifier --> EDA : parallel with
    
    DataTypeAnalyzer --> LLMModel : uses
    FeatureEngineer1 --> LLMModel : uses
    TargetVariable --> LLMModel : uses
    FeatureSelector --> LLMModel : uses
    
    LLMModel <|-- GoogleGeminiModel : implements
    LLMModel <|-- GroqModel : implements
    
    DatasetHandler --> CustomLogger : uses
    DataTypeAnalyzer --> CustomLogger : uses
    FeatureEngineer1 --> CustomLogger : uses
    TargetVariable --> CustomLogger : uses
    FeatureSelector --> CustomLogger : uses
    AutoMLClassifier --> CustomLogger : uses
    EDA --> CustomLogger : uses
    
    DatasetHandler --> AutoML_Exception : throws
    DataTypeAnalyzer --> AutoML_Exception : throws
    FeatureEngineer1 --> AutoML_Exception : throws
    TargetVariable --> AutoML_Exception : throws
    FeatureSelector --> AutoML_Exception : throws
    AutoMLClassifier --> AutoML_Exception : throws
    
    FastAPIApp --> ConfigLoader : uses
    DataTypeAnalyzer --> ConfigLoader : uses
    FeatureEngineer1 --> ConfigLoader : uses
    
    DatasetHandler --> SessionManager : uses
    FeatureEngineer1 --> SessionManager : uses
    TargetVariable --> SessionManager : uses
    FeatureSelector --> SessionManager : uses
    AutoMLClassifier --> SessionManager : uses
    EDA --> SessionManager : uses
```

### Activity Diagram

```mermaid
flowchart TD
    Start([User Uploads File]) --> A1{File Valid?}
    A1 -->|No| A2[Return Error]
    A1 -->|Yes| A3[Create Session]
    A3 --> A4[Save Raw Data]
    A4 --> A5[Initialize Data Type Analyzer]
    
    A5 --> B1[Extract Sample Data]
    B1 --> B2[Send to LLM for Analysis]
    B2 --> B3[Receive Type Recommendations]
    B3 --> B4[Generate Conversion Code]
    B4 --> B5[Apply Data Type Conversions]
    B5 --> B6[Validate Conversions]
    B6 --> B7{Conversion Success?}
    B7 -->|No| B8[Log Error & Use Original]
    B7 -->|Yes| B9[Save Converted Data]
    B8 --> C1
    B9 --> C1
    
    C1[Initialize Feature Engineer] --> C2{DateTime Columns?}
    C2 -->|Yes| C3[Extract Date Features]
    C2 -->|No| C4
    C3 --> C4{Object Columns?}
    C4 -->|Yes| C5[Send to LLM for Feature Extraction]
    C4 -->|No| C6
    C5 --> C6[Extract Weight/Duration Values]
    C6 --> C7[Combine All Features]
    C7 --> C8[Save Processed Data]
    
    C8 --> D1[Initialize Target Variable Detector]
    D1 --> D2[Analyze Problem Statement]
    D2 --> D3[Send to LLM for Target Analysis]
    D3 --> D4[Receive Target Recommendations]
    D4 --> D5[Validate Target Variable]
    D5 --> D6{Target Valid?}
    D6 -->|No| D7[Request User Input]
    D6 -->|Yes| D8[Classify Problem Type]
    D7 --> D8
    D8 --> D9[Save Target Information]
    
    D9 --> E1[Initialize Feature Selector]
    E1 --> E2[Statistical Feature Analysis]
    E2 --> E3[Correlation Analysis]
    E3 --> E4[Mutual Information Analysis]
    E4 --> E5[Chi-Square Analysis]
    E5 --> E6[Send Results to LLM]
    E6 --> E7[Receive Feature Rankings]
    E7 --> E8[Combine Selection Methods]
    E8 --> E9[Finalize Feature List]
    E9 --> E10[Save Feature Selection]
    
    E10 --> F1[Initialize ML Classifier]
    F1 --> F2[Preprocess Data]
    F2 --> F3[Label Encoding]
    F3 --> F4[MinMax Scaling]
    F4 --> F5[Split Train/Test Data]
    F5 --> F6[Initialize Models]
    
    F6 --> G1[Train Logistic Regression]
    F6 --> G2[Train Random Forest]
    F6 --> G3[Train Gradient Boosting]
    F6 --> G4[Train SVM]
    F6 --> G5[Train KNN]
    F6 --> G6[Train Decision Tree]
    
    G1 --> H1[Hyperparameter Tuning]
    G2 --> H2[Hyperparameter Tuning]
    G3 --> H3[Hyperparameter Tuning]
    G4 --> H4[Hyperparameter Tuning]
    G5 --> H5[Hyperparameter Tuning]
    G6 --> H6[Hyperparameter Tuning]
    
    H1 --> I1[Model Evaluation]
    H2 --> I2[Model Evaluation]
    H3 --> I3[Model Evaluation]
    H4 --> I4[Model Evaluation]
    H5 --> I5[Model Evaluation]
    H6 --> I6[Model Evaluation]
    
    I1 --> J1[Save Model]
    I2 --> J2[Save Model]
    I3 --> J3[Save Model]
    I4 --> J4[Save Model]
    I5 --> J5[Save Model]
    I6 --> J6[Save Model]
    
    J1 --> K1[Compile Results]
    J2 --> K1
    J3 --> K1
    J4 --> K1
    J5 --> K1
    J6 --> K1
    
    K1 --> L1[Rank Models by Performance]
    L1 --> L2[Generate EDA Report]
    L2 --> L3[Save Final Results]
    L3 --> End([Pipeline Complete])
    
    %% Parallel EDA Process
    C8 --> M1[Initialize EDA Generator]
    M1 --> M2[Load Processed Data]
    M2 --> M3[Generate Profile Report]
    M3 --> M4[Create Visualizations]
    M4 --> M5[Save HTML Report]
    M5 --> L2
    
    %% Error Handling
    A2 --> End
    
    style Start fill:#e1f5fe
    style End fill:#c8e6c9
    style A2 fill:#ffebee
    style B8 fill:#fff3e0
    style D7 fill:#fff3e0
```

### Entity Relationship Diagram (ERD)

```mermaid
erDiagram
    SESSION {
        string session_id PK
        datetime created_at
        datetime updated_at
        string status
        string user_id
        string original_filename
        int file_size
        string file_type
    }
    
    RAW_DATASET {
        string dataset_id PK
        string session_id FK
        string file_path
        int row_count
        int column_count
        json column_info
        datetime ingested_at
        string encoding
        boolean is_valid
    }
    
    DATA_TYPE_ANALYSIS {
        string analysis_id PK
        string session_id FK
        json original_types
        json recommended_types
        json conversion_code
        json llm_response
        datetime analyzed_at
        string llm_provider
        boolean applied_successfully
    }
    
    PROCESSED_DATASET {
        string processed_id PK
        string session_id FK
        string file_path
        int row_count
        int column_count
        json feature_info
        json transformations_applied
        datetime processed_at
        string processing_stage
    }
    
    FEATURE_ENGINEERING {
        string feature_eng_id PK
        string session_id FK
        json datetime_features
        json object_features
        json weight_features
        json duration_features
        json llm_generated_code
        datetime engineered_at
        int features_created
        json feature_descriptions
    }
    
    TARGET_VARIABLE {
        string target_id PK
        string session_id FK
        string problem_statement
        string target_column
        string problem_type
        json justification
        json llm_analysis
        datetime detected_at
        float confidence_score
        boolean user_confirmed
    }
    
    FEATURE_SELECTION {
        string selection_id PK
        string session_id FK
        json selected_features
        json dropped_features
        json feature_rankings
        json statistical_scores
        json llm_reasoning
        datetime selected_at
        string selection_method
        int final_feature_count
    }
    
    ML_MODEL {
        string model_id PK
        string session_id FK
        string model_name
        string model_type
        json hyperparameters
        float accuracy
        float f1_score
        json classification_report
        string model_file_path
        datetime trained_at
        json preprocessing_info
        boolean is_best_model
    }
    
    MODEL_EVALUATION {
        string evaluation_id PK
        string model_id FK
        json confusion_matrix
        json performance_metrics
        json cross_validation_scores
        json feature_importance
        datetime evaluated_at
        string evaluation_method
    }
    
    EDA_REPORT {
        string report_id PK
        string session_id FK
        string report_file_path
        json report_metadata
        json data_quality_summary
        json statistical_summary
        datetime generated_at
        int report_size_mb
    }
    
    PREPROCESSING_OBJECTS {
        string preprocess_id PK
        string session_id FK
        string object_type
        string object_file_path
        json object_metadata
        datetime created_at
        string associated_model
    }
    
    ERROR_LOG {
        string error_id PK
        string session_id FK
        string error_type
        string error_message
        json error_details
        string stack_trace
        datetime occurred_at
        string module_name
        string severity_level
    }
    
    SYSTEM_CONFIG {
        string config_id PK
        string config_type
        json config_values
        datetime updated_at
        string updated_by
        boolean is_active
    }
    
    LLM_INTERACTION {
        string interaction_id PK
        string session_id FK
        string llm_provider
        string model_name
        text prompt_text
        text response_text
        datetime interaction_at
        float response_time_ms
        int token_count
        string interaction_type
    }
    
    %% Relationships
    SESSION ||--o{ RAW_DATASET : "has"
    SESSION ||--o{ DATA_TYPE_ANALYSIS : "has"
    SESSION ||--o{ PROCESSED_DATASET : "has"
    SESSION ||--o{ FEATURE_ENGINEERING : "has"
    SESSION ||--o{ TARGET_VARIABLE : "has"
    SESSION ||--o{ FEATURE_SELECTION : "has"
    SESSION ||--o{ ML_MODEL : "has"
    SESSION ||--o{ EDA_REPORT : "has"
    SESSION ||--o{ PREPROCESSING_OBJECTS : "has"
    SESSION ||--o{ ERROR_LOG : "has"
    SESSION ||--o{ LLM_INTERACTION : "has"
    
    ML_MODEL ||--o{ MODEL_EVALUATION : "has"
    
    RAW_DATASET ||--|| DATA_TYPE_ANALYSIS : "analyzed_by"
    DATA_TYPE_ANALYSIS ||--|| PROCESSED_DATASET : "produces"
    PROCESSED_DATASET ||--|| FEATURE_ENGINEERING : "enhanced_by"
    FEATURE_ENGINEERING ||--|| TARGET_VARIABLE : "leads_to"
    TARGET_VARIABLE ||--|| FEATURE_SELECTION : "guides"
    FEATURE_SELECTION ||--|| ML_MODEL : "feeds_into"
    
    PROCESSED_DATASET ||--|| EDA_REPORT : "generates"
    ML_MODEL ||--|| PREPROCESSING_OBJECTS : "uses"
```

### Sequence Diagram - Complete AutoML Pipeline

```mermaid
sequenceDiagram
    participant User
    participant FastAPI
    participant DataHandler
    participant TypeAnalyzer
    participant LLM
    participant FeatureEng
    participant TargetDetector
    participant FeatureSelector
    participant MLClassifier
    participant EDAGenerator
    participant Storage
    
    User->>FastAPI: Upload CSV/Excel File
    FastAPI->>DataHandler: save_dataset(file)
    DataHandler->>Storage: Create session directory
    DataHandler->>Storage: Save raw data
    DataHandler-->>FastAPI: Return DataFrame + session_id
    FastAPI-->>User: Return session_id
    
    FastAPI->>TypeAnalyzer: analyze_data_type()
    TypeAnalyzer->>Storage: Load sample data
    TypeAnalyzer->>LLM: Analyze data types
    LLM-->>TypeAnalyzer: Type recommendations
    TypeAnalyzer->>TypeAnalyzer: generate_conversion_code()
    TypeAnalyzer->>TypeAnalyzer: apply_conversions()
    TypeAnalyzer->>Storage: Save converted data
    TypeAnalyzer-->>FastAPI: Conversion complete
    
    FastAPI->>FeatureEng: generate_features()
    FeatureEng->>Storage: Load converted data
    FeatureEng->>FeatureEng: process_datetime_columns()
    FeatureEng->>LLM: Generate object column features
    LLM-->>FeatureEng: Feature extraction code
    FeatureEng->>FeatureEng: extract_weight_values()
    FeatureEng->>FeatureEng: extract_duration_values()
    FeatureEng->>Storage: Save processed data
    FeatureEng-->>FastAPI: Feature engineering complete
    
    FastAPI->>TargetDetector: get_target_variable(problem_statement)
    TargetDetector->>Storage: Load processed data
    TargetDetector->>LLM: Analyze problem statement
    LLM-->>TargetDetector: Target variable + problem type
    TargetDetector->>TargetDetector: validate_target_selection()
    TargetDetector->>Storage: Save target info
    TargetDetector-->>FastAPI: Target detection complete
    
    FastAPI->>FeatureSelector: llm_response()
    FeatureSelector->>Storage: Load processed data
    FeatureSelector->>FeatureSelector: statistical_feature_selection()
    FeatureSelector->>FeatureSelector: correlation_analysis()
    FeatureSelector->>LLM: Rank features intelligently
    LLM-->>FeatureSelector: Feature rankings + reasoning
    FeatureSelector->>Storage: Save feature selection
    FeatureSelector-->>FastAPI: Feature selection complete
    
    par ML Training and EDA Generation
        FastAPI->>MLClassifier: train_models()
        MLClassifier->>Storage: Load final dataset
        MLClassifier->>MLClassifier: preprocess_data()
        loop For each algorithm
            MLClassifier->>MLClassifier: train_single_model()
            MLClassifier->>MLClassifier: hyperparameter_tuning()
            MLClassifier->>MLClassifier: evaluate_model()
            MLClassifier->>Storage: save_model()
        end
        MLClassifier->>Storage: Save results comparison
        MLClassifier-->>FastAPI: Training complete
    and
        FastAPI->>EDAGenerator: generate_report()
        EDAGenerator->>Storage: Load processed data
        EDAGenerator->>EDAGenerator: create_profile_report()
        EDAGenerator->>Storage: Save HTML report
        EDAGenerator-->>FastAPI: EDA report complete
    end
    
    FastAPI-->>User: Pipeline complete notification
    User->>FastAPI: Request results
    FastAPI->>Storage: Load all results
    Storage-->>FastAPI: Results data
    FastAPI-->>User: Return complete results
```

### Component Diagram

```mermaid
flowchart TB
    subgraph "Web Interface Layer"
        A[FastAPI Application]
        B[CORS Middleware]
        C[File Upload Handler]
        D[API Endpoints]
    end
    
    subgraph "Business Logic Layer"
        E[Data Ingestion Service]
        F[AI Analysis Service]
        G[Feature Engineering Service]
        H[ML Training Service]
        I[Report Generation Service]
    end
    
    subgraph "AI/LLM Integration Layer"
        J[LLM Provider Interface]
        K[Google Gemini Client]
        L[Groq Client]
        M[Prompt Template Engine]
        N[Response Parser]
    end
    
    subgraph "Data Processing Layer"
        O[Data Type Analyzer]
        P[Feature Engineer]
        Q[Target Variable Detector]
        R[Feature Selector]
        S[ML Classifier]
        T[EDA Generator]
    end
    
    subgraph "Storage Layer"
        U[Session Manager]
        V[File System Storage]
        W[Model Persistence]
        X[Configuration Storage]
    end
    
    subgraph "Infrastructure Layer"
        Y[Custom Logger]
        Z[Exception Handler]
        AA[Config Loader]
        BB[Environment Manager]
    end
    
    subgraph "External Dependencies"
        CC[Pandas/NumPy]
        DD[Scikit-learn]
        EE[XGBoost/LightGBM]
        FF[Plotly/Matplotlib]
        GG[ydata-profiling]
    end
    
    %% Web Layer Connections
    A --> B
    A --> C
    A --> D
    
    %% Business Logic Connections
    D --> E
    D --> F
    D --> G
    D --> H
    D --> I
    
    %% AI Integration Connections
    F --> J
    J --> K
    J --> L
    J --> M
    J --> N
    
    %% Data Processing Connections
    E --> O
    F --> O
    G --> P
    F --> Q
    F --> R
    H --> S
    I --> T
    
    %% Storage Connections
    E --> U
    G --> V
    H --> W
    AA --> X
    
    %% Infrastructure Connections
    E --> Y
    F --> Y
    G --> Y
    H --> Y
    I --> Y
    
    E --> Z
    F --> Z
    G --> Z
    H --> Z
    I --> Z
    
    A --> AA
    J --> BB
    
    %% External Dependencies
    O --> CC
    P --> CC
    S --> DD
    S --> EE
    T --> FF
    T --> GG
    
    style A fill:#e3f2fd
    style J fill:#fff3e0
    style U fill:#f3e5f5
    style Y fill:#e8f5e8
```

## Features

- **FastAPI Web Application**: RESTful API for file upload and processing
- **Data Ingestion**: Automated CSV/Excel file handling with session management
- **AI-Powered Data Type Analysis**: LLM-based intelligent data type inference and conversion
- **Automated Feature Engineering**: Complete pipeline for feature extraction and datetime processing
- **Target Variable Detection**: AI-powered identification of target variables and problem types
- **Intelligent Feature Selection**: Statistical and LLM-based feature selection for ML models
- **Automated ML Classification**: Complete classification pipeline with multiple algorithms and hyperparameter tuning
- **Exploratory Data Analysis**: Automated HTML report generation with comprehensive data insights
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
│   ├── dataCleaning/
│   │   └── featureEngineering01.py # Automated feature engineering pipeline
│   ├── problem_statement/
│   │   ├── target_variable.py      # AI-powered target variable identification
│   │   └── AutoFeatureSelector.py  # Intelligent feature selection for ML
│   ├── Classifier/
│   │   └── MLClassifier.py         # Automated ML classification with multiple algorithms
│   └── data_dashboard/
│       └── eda.py                  # Exploratory data analysis and HTML report generation
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
- **Scikit-learn**: Machine learning library and feature selection methods
- **XGBoost, LightGBM, CatBoost**: Advanced ML algorithms
- **PyCaret**: Low-code ML library
- **Matplotlib, Seaborn, Plotly**: Data visualization
- **ydata-profiling**: Automated EDA report generation
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

### Feature Engineering

```python
from src.dataCleaning.featureEngineering01 import FeatureEngineer1

# Initialize with dataset path or DataFrame
fe = FeatureEngineer1("path/to/dataset.csv")
# or
fe = FeatureEngineer1(dataframe)

# Generate features automatically
processed_df, session_id = fe.generate_features()

# Features automatically created:
# - Datetime columns → day, month, weekday, hour, minute
# - Object columns → LLM-generated feature extraction
# - Weight_value (from "15 kg" → 15.0)
# - Duration_minutes (from "2h 30min" → 150)
```

### FastAPI Web Application

```python
# Start the web server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Upload and process files via API
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_dataset.csv"
```

### Exploratory Data Analysis

```python
from src.data_dashboard.eda import EDA

# Generate comprehensive EDA report
eda = EDA(session_id="your_session_id")
html_path = eda.generate_report()

# Opens interactive HTML report with:
# - Dataset overview and statistics
# - Missing values analysis
# - Correlation matrices
# - Distribution plots
# - Data quality insights
```

### Target Variable Detection

```python
from src.problem_statement.target_variable import TargetVariable

# Initialize target variable detector
target_handler = TargetVariable(session_id="your_session_id")

# Detect target variable and problem type
result, df = target_handler.get_target_variable("Predict house prices")

print(f"Target: {result['target_variable']}")
print(f"Problem Type: {result['problem_type']}")
print(f"Justification: {result['justification']}")
```

### Feature Selection

```python
from src.problem_statement.AutoFeatureSelector import FeatureSelector

# Initialize feature selector
selector = FeatureSelector(session_id, "Predict house prices")

# Get intelligent feature selection
response = selector.llm_response()

print(f"Selected Features: {response['selected_features']}")
print(f"Dropped Features: {response['dropped_features']}")
print(f"Feature Rankings: {response['ranked_features']}")
```

### Automated ML Classification

```python
from src.Classifier.MLClassifier import AutoMLClassifier

# Initialize classifier with session and problem statement
classifier = AutoMLClassifier(
    session_id="your_session_id",
    problem_statement="Predict weather conditions",
    result=target_result,
    df=dataframe
)

# Train multiple models with hyperparameter tuning
results_df, trained_models, model_paths = classifier.train_models()

# View model performance comparison
print(results_df)
# Output:
#           Model  Accuracy  F1_Score              Best_Params
# 0  RandomForest     0.95      0.94  {'n_estimators': 100}
# 1  LogisticRegression 0.92    0.91  {'C': 1}

# Models automatically saved as .joblib files
print(f"Best model: {results_df.iloc[0]['Model']}")
print(f"Saved at: {model_paths[results_df.iloc[0]['Model']]}")
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

### Automated Feature Engineering Pipeline

- **Complete Automation**: End-to-end feature engineering from raw data to processed features
- **Datetime Processing**: Automatically extracts day, month, weekday, hour, minute from date columns
- **LLM-Powered Object Analysis**: Uses AI to generate custom feature extraction code for text columns
- **Unit Conversion**: Extracts numeric values from text with units (e.g., "15 kg" → 15.0)
- **Duration Processing**: Converts time expressions to minutes (e.g., "2h 30min" → 150)
- **Session Management**: Saves both raw and processed data with unique session IDs
- **Robust Error Handling**: Comprehensive logging and exception management

### FastAPI Web Application

- **RESTful API**: Upload CSV/Excel files and get processed results
- **CORS Support**: Cross-origin requests enabled for web frontends
- **File Validation**: Supports CSV and Excel file formats
- **JSON Response**: Returns processed data preview and session ID

### Exploratory Data Analysis

- **Automated Report Generation**: Creates comprehensive HTML reports using ydata-profiling
- **Interactive Visualizations**: Statistical summaries, correlation matrices, and distribution plots
- **Data Quality Assessment**: Missing values, duplicates, and data type analysis
- **Session Integration**: Works seamlessly with processed data from feature engineering
- **Export Capability**: Generates standalone HTML files for sharing and presentation

### Target Variable Detection

- **AI-Powered Analysis**: Uses LLM to analyze problem statements and identify target variables
- **Problem Type Classification**: Automatically determines regression, classification, or clustering
- **Justification**: Provides clear reasoning for target variable selection
- **Session Integration**: Works with processed data from feature engineering pipeline

### Intelligent Feature Selection

- **Multi-Method Selection**: Combines correlation, chi-square, mutual information, and variance analysis
- **Problem-Aware**: Adapts selection strategy based on regression, classification, or clustering
- **LLM Enhancement**: Uses AI to provide intelligent feature ranking and selection rationale
- **Statistical Foundation**: Leverages scikit-learn's feature selection methods
- **Leakage Prevention**: Identifies and removes features that may cause data leakage

### Automated ML Classification

- **Multiple Algorithms**: Supports Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, and Decision Trees
- **Hyperparameter Tuning**: Automated GridSearchCV for optimal model parameters
- **Data Preprocessing**: Automatic label encoding for categorical variables and MinMax scaling for numerical features
- **Model Persistence**: Saves trained models and preprocessing objects as .joblib files
- **Performance Metrics**: Comprehensive evaluation with accuracy, F1-score, and classification reports
- **Session Integration**: Works seamlessly with feature selection and target variable detection
- **Flexible Training**: Option to skip computationally heavy models for faster prototyping

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

### Running the Application

```bash
# Start the FastAPI web server
python app/main.py
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Run the data ingestion test
python src/datasetAnalysis/data_ingestion.py

# Run the data type analysis test
python src/datasetAnalysis/data_type_analysis.py

# Test feature engineering pipeline
python src/dataCleaning/featureEngineering01.py

# Test EDA report generation
python src/data_dashboard/eda.py

# Test target variable detection
python src/problem_statement/target_variable.py

# Test feature selection
python src/problem_statement/AutoFeatureSelector.py

# Test automated classification
python src/Classifier/MLClassifier.py

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
