
# RAG Sales Assistant

A smart AI assistant that answers questions about your sales data using RAG (Retrieval-Augmented Generation).

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Get your key from: https://makersuite.google.com/app/apikey

### 3. Run the Application

**Start Backend** (Terminal 1):

```bash
python main.py
```

**Start Frontend** (Terminal 2):

```bash
streamlit run streamlit_app_client.py
```

Open your browser to `http://localhost:8501`

## How to Use

1. Upload your CSV or Excel files
2. Click "Process Documents"
3. Ask questions about your data!

### Example Questions

* "What is the total sales revenue?"
* "Which region has the highest sales?"
* "Compare North vs South region"
* "Show me all products"

## API Documentation

View interactive API docs at: `http://localhost:8000/docs`

## Testing

```bash
python test_api.py path/to/your/data.csv
```

## Troubleshooting

**API not connecting?**

* Make sure `python main.py` is running
* Check port 8000 is available

**Google API errors?**

* Verify your API key in `.env`
* Check you have API credits

## Files

* `main.py` - FastAPI backend
* `streamlit_app_client.py` - Streamlit UI
* `rag-system.py` - Standalone version (no API)
* `test_api.py` - Test script

## License

MIT
