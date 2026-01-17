# 🏎️ F1 Race Strategy Analyzer

An AI-powered Formula 1 race strategy analysis system that combines TensorFlow machine learning models, vector databases, and large language models to generate data-driven race strategies and predictions.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

### 🤖 AI/ML Capabilities
- **Race Outcome Prediction**: LSTM-based model predicting finishing positions
- **Pit Stop Optimization**: Deep Q-Network for optimal pit strategy
- **Tire Degradation**: Regression model for lap time prediction
- **Semantic Search**: Vector database for historical strategy retrieval
- **Strategy Generation**: Claude AI for natural language race strategies

### ⚡ Event-Driven Architecture
- Automatic data ingestion when races end
- Real-time strategy updates
- Weekly data refresh
- SNS notifications for updates

### 📊 Data Sources
- **Ergast API**: Historical race data (1950-2024)
- **FastF1**: Detailed telemetry and timing data
- **OpenF1 API**: Real-time race data

### 🚀 REST API
- Race data endpoints
- Strategy generation
- Predictions and forecasts
- Semantic query interface
- What-if scenario analysis

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Data APIs  │────▶│   Lambda     │────▶│  DynamoDB   │
│  (Ergast,   │     │  Functions   │     │   (Metadata)│
│   FastF1)   │     └──────────────┘     └─────────────┘
└─────────────┘            │                     │
                          │                     │
                          ▼                     ▼
                   ┌──────────────┐      ┌─────────────┐
                   │  S3 Bucket   │      │   FastAPI   │
                   │  (Raw Data)  │      │   Backend   │
                   └──────────────┘      └─────────────┘
                          │                     │
                          ▼                     ▼
        ┌─────────────────────────────────────────────┐
        │          AI/ML Processing Layer             │
        ├─────────────────────────────────────────────┤
        │  ┌────────────┐  ┌──────────┐  ┌─────────┐ │
        │  │ TensorFlow │  │ Pinecone │  │ Claude  │ │
        │  │   Models   │  │ Vector DB│  │   API   │ │
        │  └────────────┘  └──────────┘  └─────────┘ │
        └─────────────────────────────────────────────┘
```

## 🚦 Quick Start

Choose your preferred setup method:

### Option 1: Local Development (Recommended for Learning)

```bash
# 1. Clone/create directory
mkdir f1-race-analyzer && cd f1-race-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cat > .env << EOF
PINECONE_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
EOF

# 4. Initialize and collect data
python << EOF
from vector_database import F1VectorDatabase
from data_collection import F1DataCollector
import os

# Setup vector DB
vdb = F1VectorDatabase(api_key=os.getenv('PINECONE_API_KEY'))
vdb.create_index()

# Collect sample data
collector = F1DataCollector()
seasons = collector.get_seasons_data(2024, 2024)
print(f"Collected {len(seasons[0]['races'])} races!")
EOF

# 5. Start the API
python fastapi_backend.py
```

Visit http://localhost:8000/docs for interactive API documentation!

### Option 2: Docker (Fastest Setup)

```bash
# 1. Create .env file with your API keys
cat > .env << EOF
PINECONE_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
EOF

# 2. Start all services
docker-compose up -d

# 3. View logs
docker-compose logs -f api
```

### Option 3: Full AWS Deployment

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for complete AWS setup.

## 📚 Documentation

- **[QUICKSTART.md](./QUICKSTART.md)** - Get up and running in 15 minutes
- **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** - Full production deployment
- **[Architecture Document](./f1-strategy-analyzer-architecture.md)** - System design details

## 🎯 Usage Examples

### Generate a Race Strategy

```python
from llm_strategy_generator import F1StrategyGenerator
import os

generator = F1StrategyGenerator(api_key=os.getenv('ANTHROPIC_API_KEY'))

race_info = {
    'circuit': 'Monaco',
    'weather_forecast': 'Dry, 24°C',
    'total_laps': 78
}

strategy = generator.generate_race_strategy(
    race_info=race_info,
    historical_context=[],
    ml_predictions=None
)

print(strategy['executive_summary'])
```

### Predict Race Outcomes

```python
from tensorflow_models import RaceOutcomePredictor
import numpy as np

# Load trained model
predictor = RaceOutcomePredictor(sequence_length=10, num_features=30)
predictor.load_model('models/race_predictor.h5')

# Make prediction
race_sequence = np.random.randn(1, 10, 30)  # Your feature data
position, probabilities = predictor.predict_position(race_sequence)

print(f"Predicted position: {position}")
print(f"Top 3 probability: {probabilities[:3].sum():.2%}")
```

### Semantic Search

```python
from vector_database import F1VectorDatabase
import os

vector_db = F1VectorDatabase(api_key=os.getenv('PINECONE_API_KEY'))
vector_db.create_index()

results = vector_db.search_similar_strategies(
    query="What strategies work best at Monaco in wet conditions?",
    top_k=5
)

for result in results:
    print(f"Race: {result['metadata']['race_name']}")
    print(f"Strategy: {result['metadata']['description']}")
    print(f"Similarity: {result['score']:.2%}\n")
```

### REST API Examples

```bash
# Get race predictions
curl http://localhost:8000/race/2024_monaco_gp/predictions

# Generate strategy
curl -X POST http://localhost:8000/strategy/generate \
  -H "Content-Type: application/json" \
  -d '{
    "race_info": {
      "season": 2025,
      "round": 8,
      "circuit": "monaco",
      "weather_forecast": "Dry, 24C"
    }
  }'

# Semantic query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Best tire strategies for Silverstone",
    "circuit": "silverstone"
  }'
```

## 🧪 Training Models

Train all models on historical data:

```bash
python train_models.py
```

This will:
1. Collect 10 years of historical F1 data
2. Engineer features for ML models
3. Train race outcome predictor (LSTM)
4. Train tire degradation model
5. Initialize pit stop optimizer
6. Save trained models to `./models/`

Training takes ~30-60 minutes depending on your hardware.

## 📊 Project Structure

```
f1-race-analyzer/
├── data_collection.py           # F1 data scraping (Ergast, FastF1)
├── feature_engineering.py       # ML feature preparation
├── tensorflow_models.py         # AI models (LSTM, DQN, Regression)
├── vector_database.py           # Pinecone integration
├── llm_strategy_generator.py    # Claude AI integration
├── lambda_functions.py          # AWS Lambda handlers
├── fastapi_backend.py           # REST API server
├── train_models.py              # Model training script
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Multi-container setup
├── .env.example                 # Environment variables template
├── QUICKSTART.md                # Quick start guide
├── DEPLOYMENT_GUIDE.md          # Full deployment docs
└── README.md                    # This file
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | TensorFlow 2.15 |
| **Vector DB** | Pinecone |
| **LLM** | Claude (Anthropic) |
| **API Framework** | FastAPI |
| **Cloud Services** | AWS (Lambda, S3, DynamoDB, EventBridge, SNS) |
| **Data Sources** | Ergast API, FastF1, OpenF1 |
| **Containerization** | Docker, docker-compose |
| **Language** | Python 3.9+ |

## 🎓 Learning Outcomes

This project teaches you:

### Machine Learning
✅ LSTM networks for sequential prediction  
✅ Deep Q-Networks for reinforcement learning  
✅ Regression models for time series  
✅ Feature engineering for motorsport data  
✅ Model training, evaluation, and deployment  

### AI Engineering
✅ Vector databases and semantic search  
✅ RAG (Retrieval Augmented Generation)  
✅ LLM integration and prompt engineering  
✅ Embedding generation and similarity search  

### Software Engineering
✅ REST API development with FastAPI  
✅ Event-driven architecture  
✅ Serverless computing (AWS Lambda)  
✅ Cloud infrastructure (S3, DynamoDB)  
✅ Docker containerization  
✅ Data pipeline design  

## 🔮 Future Enhancements

- [ ] Real-time race monitoring during live events
- [ ] React/Next.js dashboard with visualizations
- [ ] Mobile app (React Native)
- [ ] Multi-language support for strategies
- [ ] Driver performance analysis
- [ ] Team comparison tools
- [ ] Championship prediction models
- [ ] Weather integration with live forecasts
- [ ] Social features (share strategies, discuss races)
- [ ] GraphQL API
- [ ] WebSocket support for live updates

## 💰 Cost Estimate

### Development/Testing (Monthly)
- AWS Free Tier: $0
- Pinecone (Starter): $70
- Claude API: $20-50
- **Total: ~$90-120/month**

### Production (Monthly)
- AWS Services: $50-100
- Pinecone (Standard): $70-200
- Claude API: $100-200
- **Total: ~$220-500/month**

## 🤝 Contributing

Contributions are welcome! Areas where you can help:

1. **Data Sources**: Add more F1 data providers
2. **Models**: Improve model architectures
3. **Features**: Add new analysis features
4. **Documentation**: Improve guides and examples
5. **Testing**: Add unit and integration tests
6. **UI**: Build frontend dashboards

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ergast API** - Historical F1 data
- **FastF1** - Python library for F1 telemetry
- **Anthropic** - Claude AI
- **Pinecone** - Vector database
- **F1 Community** - Data and insights

## 📧 Contact

Questions? Feedback? Open an issue or reach out!

---

**Built with ❤️ for F1 fans and AI enthusiasts**

*Learn AI/ML technologies while analyzing the world's most exciting motorsport!*
