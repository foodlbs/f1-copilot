"""
F1 AWS Lambda Functions Module

Serverless functions for the F1 Race Strategy Analyzer:
- Data ingestion when races end
- Scheduled data refresh
- Strategy generation on demand
- SNS notifications

Author: F1 Race Strategy Analyzer
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients (initialized lazily)
_s3_client = None
_dynamodb = None
_sns_client = None


def get_s3_client():
    """Get or create S3 client"""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3')
    return _s3_client


def get_dynamodb():
    """Get or create DynamoDB resource"""
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource('dynamodb')
    return _dynamodb


def get_sns_client():
    """Get or create SNS client"""
    global _sns_client
    if _sns_client is None:
        _sns_client = boto3.client('sns')
    return _sns_client


# Environment variables
S3_BUCKET = os.environ.get('F1_DATA_BUCKET', 'f1-race-analyzer-data')
DYNAMODB_TABLE = os.environ.get('F1_METADATA_TABLE', 'f1-race-metadata')
SNS_TOPIC_ARN = os.environ.get('F1_SNS_TOPIC', '')


class F1DataManager:
    """Manages F1 data in AWS services"""
    
    def __init__(self):
        self.s3 = get_s3_client()
        self.dynamodb = get_dynamodb()
        self.table = self.dynamodb.Table(DYNAMODB_TABLE)
    
    def store_race_data(self, race_data: Dict) -> bool:
        """Store race data in S3 and metadata in DynamoDB"""
        try:
            season = race_data.get('season', 0)
            round_num = race_data.get('round', 0)
            race_id = f"{season}_{round_num}"
            
            # Store full data in S3
            s3_key = f"races/{season}/{race_id}.json"
            self.s3.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=json.dumps(race_data, default=str),
                ContentType='application/json'
            )
            
            # Store metadata in DynamoDB
            metadata = {
                'race_id': race_id,
                'season': season,
                'round': round_num,
                'race_name': race_data.get('race_name', ''),
                'circuit_id': race_data.get('circuit_id', ''),
                'date': race_data.get('date', ''),
                's3_key': s3_key,
                'updated_at': datetime.utcnow().isoformat(),
                'winner': race_data.get('results', [{}])[0].get('driver_id', ''),
                'num_finishers': len([r for r in race_data.get('results', []) 
                                     if r.get('status') == 'Finished'])
            }
            
            self.table.put_item(Item=metadata)
            
            logger.info(f"Stored race data: {race_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store race data: {e}")
            return False
    
    def get_race_data(self, season: int, round_num: int) -> Optional[Dict]:
        """Retrieve race data from S3"""
        try:
            s3_key = f"races/{season}/{season}_{round_num}.json"
            response = self.s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise
    
    def list_races(self, season: Optional[int] = None) -> List[Dict]:
        """List available races from DynamoDB"""
        try:
            if season:
                response = self.table.query(
                    IndexName='season-index',
                    KeyConditionExpression=boto3.dynamodb.conditions.Key('season').eq(season)
                )
            else:
                response = self.table.scan()
            
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Failed to list races: {e}")
            return []
    
    def get_latest_race(self) -> Optional[Dict]:
        """Get the most recently added race"""
        try:
            response = self.table.scan(
                Limit=1,
                ScanIndexForward=False
            )
            items = response.get('Items', [])
            return items[0] if items else None
        except Exception as e:
            logger.error(f"Failed to get latest race: {e}")
            return None


def send_notification(subject: str, message: str):
    """Send SNS notification"""
    if not SNS_TOPIC_ARN:
        logger.warning("SNS topic not configured")
        return
    
    try:
        sns = get_sns_client()
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )
        logger.info(f"Sent notification: {subject}")
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")


# =============================================================================
# Lambda Handlers
# =============================================================================

def race_data_ingestion_handler(event: Dict, context: Any) -> Dict:
    """
    Lambda handler for race data ingestion.
    
    Triggered by:
    - EventBridge rule after race ends
    - Manual API Gateway invocation
    
    Event format:
    {
        "season": 2024,
        "round": 5,
        "force_refresh": false
    }
    """
    logger.info(f"Race data ingestion triggered: {event}")
    
    try:
        season = event.get('season', datetime.now().year)
        round_num = event.get('round')
        force_refresh = event.get('force_refresh', False)
        
        # Import here to avoid cold start issues
        from data_collection import F1DataCollector
        
        collector = F1DataCollector(use_fastf1=False)
        data_manager = F1DataManager()
        
        if round_num:
            # Ingest specific race
            logger.info(f"Collecting data for {season} Round {round_num}")
            
            # Check if already exists
            if not force_refresh:
                existing = data_manager.get_race_data(season, round_num)
                if existing:
                    return {
                        'statusCode': 200,
                        'body': json.dumps({
                            'message': 'Race data already exists',
                            'race_id': f"{season}_{round_num}"
                        })
                    }
            
            race_data = collector.collect_race_data(season, round_num)
            data_manager.store_race_data(race_data.__dict__ if hasattr(race_data, '__dict__') else race_data)
            
            send_notification(
                f"F1 Data Updated: {season} Round {round_num}",
                f"Race data for {race_data.race_name} has been ingested."
            )
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Race data ingested successfully',
                    'race_id': f"{season}_{round_num}",
                    'race_name': race_data.race_name
                })
            }
        else:
            # Get schedule and find races to ingest
            schedule = collector.ergast.get_season_schedule(season)
            ingested = []
            
            for race in schedule:
                race_date = datetime.strptime(race['date'], '%Y-%m-%d')
                if race_date < datetime.now():
                    round_n = int(race['round'])
                    
                    if not force_refresh:
                        existing = data_manager.get_race_data(season, round_n)
                        if existing:
                            continue
                    
                    try:
                        race_data = collector.collect_race_data(season, round_n)
                        data_manager.store_race_data(
                            race_data.__dict__ if hasattr(race_data, '__dict__') else race_data
                        )
                        ingested.append(f"{season}_{round_n}")
                    except Exception as e:
                        logger.warning(f"Failed to ingest {season} R{round_n}: {e}")
            
            if ingested:
                send_notification(
                    f"F1 Data Batch Update: {season}",
                    f"Ingested {len(ingested)} races: {', '.join(ingested)}"
                )
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'Ingested {len(ingested)} races',
                    'races': ingested
                })
            }
            
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def weekly_refresh_handler(event: Dict, context: Any) -> Dict:
    """
    Lambda handler for weekly data refresh.
    
    Triggered by:
    - EventBridge scheduled rule (weekly)
    
    Actions:
    - Check for new race data
    - Update vector database
    - Retrain models if needed
    """
    logger.info("Weekly refresh triggered")
    
    try:
        current_year = datetime.now().year
        
        # Trigger data ingestion
        ingestion_result = race_data_ingestion_handler({
            'season': current_year,
            'force_refresh': False
        }, context)
        
        # Update vector database
        try:
            from vector_database import F1VectorDatabase, StrategyIndexer
            
            db = F1VectorDatabase()
            db.create_index()
            
            # Get recent races
            data_manager = F1DataManager()
            recent_races = data_manager.list_races(current_year)
            
            if recent_races:
                indexer = StrategyIndexer(db)
                # Load full race data for each
                races_data = []
                for race_meta in recent_races[:5]:  # Last 5 races
                    race_data = data_manager.get_race_data(
                        race_meta['season'],
                        race_meta['round']
                    )
                    if race_data:
                        races_data.append(race_data)
                
                indexed_count = indexer.index_race_strategies(races_data)
                logger.info(f"Indexed {indexed_count} strategies")
        except Exception as e:
            logger.warning(f"Vector DB update failed: {e}")
        
        send_notification(
            "F1 Analyzer Weekly Refresh Complete",
            f"Refresh completed at {datetime.utcnow().isoformat()}"
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Weekly refresh completed',
                'ingestion': json.loads(ingestion_result.get('body', '{}'))
            })
        }
        
    except Exception as e:
        logger.error(f"Weekly refresh failed: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def strategy_generation_handler(event: Dict, context: Any) -> Dict:
    """
    Lambda handler for on-demand strategy generation.
    
    Triggered by:
    - API Gateway POST request
    
    Event format (from API Gateway):
    {
        "body": {
            "circuit": "monaco",
            "weather_forecast": "Dry, 24C",
            "total_laps": 78,
            "driver": "Max Verstappen",
            "grid_position": 1
        }
    }
    """
    logger.info(f"Strategy generation triggered: {event}")
    
    try:
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)
        
        # Import strategy generator
        from llm_strategy_generator import F1StrategyGenerator, StrategyRequest
        from vector_database import F1VectorDatabase
        
        # Initialize components
        generator = F1StrategyGenerator()
        
        # Get historical context
        historical_context = []
        try:
            vector_db = F1VectorDatabase()
            vector_db.create_index()
            query = f"Race strategy for {body.get('circuit', 'unknown')}"
            historical_context = vector_db.search_similar_strategies(query, top_k=5)
        except Exception as e:
            logger.warning(f"Could not get historical context: {e}")
        
        # Generate strategy
        race_info = {
            'circuit': body.get('circuit', 'Unknown'),
            'weather_forecast': body.get('weather_forecast', 'Unknown'),
            'total_laps': body.get('total_laps', 50),
            'driver': body.get('driver'),
            'constructor': body.get('constructor'),
            'grid_position': body.get('grid_position')
        }
        
        strategy = generator.generate_race_strategy(
            race_info=race_info,
            historical_context=historical_context
        )
        
        response_body = {
            'executive_summary': strategy.executive_summary,
            'recommended_strategy': strategy.recommended_strategy,
            'alternative_strategies': strategy.alternative_strategies,
            'risk_assessment': strategy.risk_assessment,
            'weather_contingency': strategy.weather_contingency,
            'confidence_score': strategy.confidence_score
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_body, default=str)
        }
        
    except Exception as e:
        logger.error(f"Strategy generation failed: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }


def prediction_handler(event: Dict, context: Any) -> Dict:
    """
    Lambda handler for ML predictions.
    
    Triggered by:
    - API Gateway GET/POST request
    
    Event format:
    {
        "race_id": "2024_5",
        "driver_id": "max_verstappen",
        "prediction_type": "position" | "pit_stop" | "tire_degradation"
    }
    """
    logger.info(f"Prediction requested: {event}")
    
    try:
        # Parse request
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)
        
        prediction_type = body.get('prediction_type', 'position')
        
        # For now, return placeholder predictions
        # In production, this would load and run the ML models
        
        if prediction_type == 'position':
            prediction = {
                'predicted_position': 3,
                'confidence': 0.72,
                'position_probabilities': {
                    'P1': 0.15,
                    'P2': 0.22,
                    'P3': 0.35,
                    'P4': 0.18,
                    'P5': 0.10
                }
            }
        elif prediction_type == 'pit_stop':
            prediction = {
                'recommended_action': 'Stay out',
                'confidence': 0.85,
                'reasoning': 'Tire performance still optimal'
            }
        elif prediction_type == 'tire_degradation':
            prediction = {
                'current_degradation': 0.02,
                'predicted_lap_time_loss': 0.15,
                'recommended_stint_length': 25
            }
        else:
            prediction = {'error': 'Unknown prediction type'}
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(prediction)
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def semantic_query_handler(event: Dict, context: Any) -> Dict:
    """
    Lambda handler for semantic search queries.
    
    Triggered by:
    - API Gateway POST request
    
    Event format:
    {
        "query": "Best tire strategies for Silverstone",
        "circuit": "silverstone",  # optional filter
        "top_k": 5
    }
    """
    logger.info(f"Semantic query: {event}")
    
    try:
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)
        
        query = body.get('query', '')
        circuit = body.get('circuit')
        top_k = body.get('top_k', 5)
        
        from vector_database import F1VectorDatabase
        
        db = F1VectorDatabase()
        db.create_index()
        
        filter_dict = {'circuit_id': circuit} if circuit else None
        
        results = db.search_similar_strategies(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'query': query,
                'results': results,
                'count': len(results)
            }, default=str)
        }
        
    except Exception as e:
        logger.error(f"Semantic query failed: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def health_check_handler(event: Dict, context: Any) -> Dict:
    """Simple health check endpoint"""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'f1-race-analyzer'
        })
    }


# =============================================================================
# Local Testing
# =============================================================================

if __name__ == "__main__":
    # Test handlers locally
    print("Testing Lambda handlers locally...")
    
    # Test health check
    result = health_check_handler({}, None)
    print(f"\nHealth check: {result['body']}")
    
    # Test strategy generation (mock)
    event = {
        'body': {
            'circuit': 'monaco',
            'weather_forecast': 'Dry, 24C',
            'total_laps': 78
        }
    }
    
    print("\nTesting strategy generation...")
    # Note: Uncomment below to test (requires API keys)
    # result = strategy_generation_handler(event, None)
    # print(f"Strategy result: {result['body'][:200]}...")
    
    print("\n✅ Lambda handlers ready for deployment")
