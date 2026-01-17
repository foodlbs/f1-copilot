"""
F1 LLM Strategy Generator Module

Uses Claude AI (Anthropic) to generate natural language race strategies
with RAG (Retrieval Augmented Generation) from historical data.

Author: F1 Race Strategy Analyzer
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic not installed. LLM features limited.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyRequest:
    """Request for strategy generation"""
    circuit: str
    weather_forecast: str
    total_laps: int
    driver: Optional[str] = None
    constructor: Optional[str] = None
    grid_position: Optional[int] = None
    tire_compounds_available: List[str] = None
    safety_car_probability: Optional[float] = None
    
    def __post_init__(self):
        if self.tire_compounds_available is None:
            self.tire_compounds_available = ['SOFT', 'MEDIUM', 'HARD']


@dataclass
class StrategyResponse:
    """Generated strategy response"""
    executive_summary: str
    recommended_strategy: Dict[str, Any]
    alternative_strategies: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    weather_contingency: str
    key_decision_points: List[Dict[str, Any]]
    confidence_score: float
    reasoning: str


class PromptTemplates:
    """Templates for LLM prompts"""
    
    SYSTEM_PROMPT = """You are an expert Formula 1 race strategist with decades of experience.
Your role is to analyze race conditions and provide detailed, actionable pit stop strategies.

Key principles:
1. Always consider tire degradation curves for each compound
2. Account for track position and the "dirty air" effect
3. Factor in weather conditions and their impact on tire performance
4. Consider the competitor strategies when making recommendations
5. Balance risk vs reward based on championship standings
6. Account for circuit-specific characteristics (overtaking difficulty, pit lane time loss)

Provide strategies in a structured format with clear reasoning."""

    STRATEGY_PROMPT = """Generate a comprehensive race strategy for the following scenario:

## Race Information
- Circuit: {circuit}
- Total Laps: {total_laps}
- Weather Forecast: {weather_forecast}
{driver_info}
{grid_info}
- Available Tire Compounds: {tire_compounds}
{safety_car_info}

## Historical Context
{historical_context}

## ML Model Predictions
{ml_predictions}

## Request
Provide a detailed race strategy including:

1. **Executive Summary**: A 2-3 sentence overview of the recommended approach
2. **Primary Strategy**: 
   - Number of pit stops
   - Target lap windows for each stop
   - Tire compound sequence
   - Reasoning for this approach

3. **Alternative Strategies**:
   - At least 2 backup strategies
   - When to switch to each alternative
   - Risk/reward analysis

4. **Risk Assessment**:
   - Key risks to the strategy
   - Mitigation approaches
   - Probability estimates

5. **Weather Contingency**:
   - What to do if weather changes
   - Trigger points for strategy changes

6. **Key Decision Points**:
   - Critical laps where strategy could change
   - What to monitor at each point

Format your response as a structured JSON object."""

    WHAT_IF_PROMPT = """Analyze the following "what-if" scenario for an F1 race:

## Base Scenario
- Circuit: {circuit}
- Current Lap: {current_lap} of {total_laps}
- Current Position: P{position}
- Current Tire: {current_tire} (Age: {tire_age} laps)
- Pit Stops So Far: {pit_stops_done}

## What-If Scenario
{scenario_description}

## Question
{question}

Provide a detailed analysis of:
1. How this scenario would affect the current strategy
2. Recommended adjustments
3. Expected outcome comparison
4. Key risks and opportunities

Format as structured JSON."""

    RACE_ANALYSIS_PROMPT = """Analyze the following completed race and extract strategic insights:

## Race Details
{race_details}

## Results Summary
{results_summary}

## Pit Stop Data
{pit_stop_data}

Provide analysis including:
1. Key strategic decisions that affected the outcome
2. What worked well
3. What could have been done differently
4. Lessons for future races at this circuit

Format as structured JSON."""


class F1StrategyGenerator:
    """
    LLM-powered F1 race strategy generator using Claude AI.
    
    Combines historical data (via RAG) with ML predictions to
    generate comprehensive race strategies.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        self.client = None
        
        if ANTHROPIC_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"Anthropic client initialized with model: {model}")
        else:
            logger.warning("Anthropic client not available")
        
        self.templates = PromptTemplates()
        self.conversation_history = []
    
    def _call_claude(self, 
                     user_prompt: str,
                     system_prompt: Optional[str] = None,
                     max_tokens: int = 4000,
                     temperature: float = 0.7) -> str:
        """Make a call to Claude API"""
        if not self.client:
            return self._generate_fallback_response(user_prompt)
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or self.templates.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._generate_fallback_response(user_prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a fallback response when API is unavailable"""
        return json.dumps({
            "executive_summary": "API unavailable. Please configure ANTHROPIC_API_KEY.",
            "recommended_strategy": {
                "stops": 2,
                "description": "Default 2-stop strategy",
                "pit_windows": ["Lap 15-20", "Lap 35-40"],
                "compound_sequence": ["MEDIUM", "HARD", "SOFT"]
            },
            "alternative_strategies": [],
            "risk_assessment": {"note": "Using fallback strategy"},
            "weather_contingency": "Monitor conditions",
            "key_decision_points": [],
            "confidence_score": 0.5,
            "reasoning": "Fallback strategy generated without AI assistance"
        })
    
    def generate_race_strategy(self,
                               race_info: Dict[str, Any],
                               historical_context: List[Dict] = None,
                               ml_predictions: Dict = None) -> StrategyResponse:
        """
        Generate a comprehensive race strategy.
        
        Args:
            race_info: Race details (circuit, weather, laps, etc.)
            historical_context: Similar past strategies from vector DB
            ml_predictions: Predictions from ML models
        
        Returns:
            StrategyResponse with complete strategy details
        """
        # Build prompt
        driver_info = ""
        if race_info.get('driver'):
            driver_info = f"- Driver: {race_info['driver']}"
            if race_info.get('constructor'):
                driver_info += f" ({race_info['constructor']})"
        
        grid_info = ""
        if race_info.get('grid_position'):
            grid_info = f"- Starting Position: P{race_info['grid_position']}"
        
        safety_car_info = ""
        if race_info.get('safety_car_probability'):
            safety_car_info = f"- Safety Car Probability: {race_info['safety_car_probability']*100:.0f}%"
        
        # Format historical context
        historical_str = "No historical data available."
        if historical_context:
            historical_items = []
            for ctx in historical_context[:5]:
                meta = ctx.get('metadata', {})
                historical_items.append(
                    f"- {meta.get('season', '?')} {meta.get('race_name', '?')}: "
                    f"{meta.get('text', 'No details')}"
                )
            historical_str = "\n".join(historical_items)
        
        # Format ML predictions
        ml_str = "No ML predictions available."
        if ml_predictions:
            ml_items = []
            if 'predicted_position' in ml_predictions:
                ml_items.append(f"- Predicted finish: P{ml_predictions['predicted_position']}")
            if 'pit_recommendation' in ml_predictions:
                rec = ml_predictions['pit_recommendation']
                ml_items.append(f"- Pit recommendation: {rec.get('recommended_action', 'N/A')}")
            if 'tire_analysis' in ml_predictions:
                for compound, data in ml_predictions['tire_analysis'].items():
                    ml_items.append(
                        f"- {compound}: Optimal stint ~{data.get('recommended_stint_length', '?')} laps"
                    )
            ml_str = "\n".join(ml_items) if ml_items else ml_str
        
        prompt = self.templates.STRATEGY_PROMPT.format(
            circuit=race_info.get('circuit', 'Unknown'),
            total_laps=race_info.get('total_laps', 50),
            weather_forecast=race_info.get('weather_forecast', 'Unknown'),
            driver_info=driver_info,
            grid_info=grid_info,
            tire_compounds=', '.join(race_info.get('tire_compounds', ['SOFT', 'MEDIUM', 'HARD'])),
            safety_car_info=safety_car_info,
            historical_context=historical_str,
            ml_predictions=ml_str
        )
        
        # Call Claude
        response_text = self._call_claude(prompt, temperature=0.7)
        
        # Parse response
        try:
            # Try to extract JSON from response
            response_data = self._extract_json(response_text)
        except:
            response_data = {
                "executive_summary": response_text[:500],
                "recommended_strategy": {"description": "See full response"},
                "alternative_strategies": [],
                "risk_assessment": {},
                "weather_contingency": "",
                "key_decision_points": [],
                "confidence_score": 0.7,
                "reasoning": response_text
            }
        
        return StrategyResponse(
            executive_summary=response_data.get('executive_summary', ''),
            recommended_strategy=response_data.get('recommended_strategy', {}),
            alternative_strategies=response_data.get('alternative_strategies', []),
            risk_assessment=response_data.get('risk_assessment', {}),
            weather_contingency=response_data.get('weather_contingency', ''),
            key_decision_points=response_data.get('key_decision_points', []),
            confidence_score=response_data.get('confidence_score', 0.7),
            reasoning=response_data.get('reasoning', '')
        )
    
    def analyze_what_if(self,
                        base_scenario: Dict[str, Any],
                        what_if_description: str,
                        question: str) -> Dict[str, Any]:
        """
        Analyze a what-if scenario.
        
        Args:
            base_scenario: Current race state
            what_if_description: Description of the hypothetical change
            question: Specific question to answer
        
        Returns:
            Analysis of the scenario
        """
        prompt = self.templates.WHAT_IF_PROMPT.format(
            circuit=base_scenario.get('circuit', 'Unknown'),
            current_lap=base_scenario.get('current_lap', 1),
            total_laps=base_scenario.get('total_laps', 50),
            position=base_scenario.get('position', 10),
            current_tire=base_scenario.get('current_tire', 'MEDIUM'),
            tire_age=base_scenario.get('tire_age', 0),
            pit_stops_done=base_scenario.get('pit_stops_done', 0),
            scenario_description=what_if_description,
            question=question
        )
        
        response_text = self._call_claude(prompt, temperature=0.5)
        
        try:
            return self._extract_json(response_text)
        except:
            return {
                "analysis": response_text,
                "recommendation": "See full analysis",
                "confidence": 0.6
            }
    
    def analyze_completed_race(self, race_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a completed race for strategic insights.
        
        Args:
            race_data: Complete race data with results and pit stops
        
        Returns:
            Strategic analysis and insights
        """
        # Format race details
        race_details = (
            f"Race: {race_data.get('race_name', 'Unknown')}\n"
            f"Season: {race_data.get('season', 'Unknown')}\n"
            f"Circuit: {race_data.get('circuit_id', 'Unknown')}\n"
            f"Total Laps: {race_data.get('total_laps', '?')}\n"
            f"Weather: {json.dumps(race_data.get('weather', {}))}"
        )
        
        # Format results
        results = race_data.get('results', [])[:10]
        results_lines = []
        for r in results:
            results_lines.append(
                f"P{r.get('position', '?')}: {r.get('driver_name', r.get('driver_id', '?'))} "
                f"(Grid: P{r.get('grid', '?')}, Status: {r.get('status', '?')})"
            )
        results_summary = "\n".join(results_lines)
        
        # Format pit stops
        pit_stops = race_data.get('pit_stops', [])
        pit_by_driver = {}
        for ps in pit_stops:
            driver = ps.get('driver_id', 'Unknown')
            if driver not in pit_by_driver:
                pit_by_driver[driver] = []
            pit_by_driver[driver].append(f"Lap {ps.get('lap', '?')}: {ps.get('duration', '?')}s")
        
        pit_lines = []
        for driver, stops in list(pit_by_driver.items())[:10]:
            pit_lines.append(f"{driver}: {', '.join(stops)}")
        pit_stop_data = "\n".join(pit_lines)
        
        prompt = self.templates.RACE_ANALYSIS_PROMPT.format(
            race_details=race_details,
            results_summary=results_summary,
            pit_stop_data=pit_stop_data
        )
        
        response_text = self._call_claude(prompt, temperature=0.5)
        
        try:
            return self._extract_json(response_text)
        except:
            return {
                "analysis": response_text,
                "key_insights": [],
                "lessons_learned": []
            }
    
    def compare_strategies(self,
                           strategy_a: Dict[str, Any],
                           strategy_b: Dict[str, Any],
                           race_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two different strategies"""
        prompt = f"""Compare these two F1 race strategies:

## Strategy A
{json.dumps(strategy_a, indent=2)}

## Strategy B
{json.dumps(strategy_b, indent=2)}

## Race Context
Circuit: {race_context.get('circuit', 'Unknown')}
Laps: {race_context.get('total_laps', 50)}
Weather: {race_context.get('weather', 'Unknown')}

Provide:
1. Pros and cons of each strategy
2. Which is better for different scenarios
3. Overall recommendation
4. Confidence level

Format as JSON."""

        response_text = self._call_claude(prompt, temperature=0.5)
        
        try:
            return self._extract_json(response_text)
        except:
            return {"comparison": response_text}
    
    def explain_strategy_decision(self,
                                   decision: str,
                                   context: Dict[str, Any]) -> str:
        """Get a natural language explanation for a strategy decision"""
        prompt = f"""Explain the following F1 strategy decision in simple terms:

Decision: {decision}

Context:
- Lap: {context.get('lap', '?')} of {context.get('total_laps', '?')}
- Position: P{context.get('position', '?')}
- Tire: {context.get('tire', 'Unknown')} ({context.get('tire_age', '?')} laps old)
- Weather: {context.get('weather', 'Unknown')}

Provide a 2-3 sentence explanation that a casual F1 fan could understand."""

        return self._call_claude(prompt, max_tokens=500, temperature=0.7)
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from response text"""
        # Try to find JSON in the text
        import re
        
        # Look for JSON block
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Try parsing the entire text as JSON
        try:
            return json.loads(text)
        except:
            pass
        
        # Look for JSON-like structure
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        raise ValueError("Could not extract JSON from response")
    
    def chat(self, message: str) -> str:
        """
        Have a conversation about F1 strategy.
        
        Args:
            message: User message
        
        Returns:
            AI response
        """
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Build messages for Claude
        if self.client:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.7,
                    system=self.templates.SYSTEM_PROMPT,
                    messages=self.conversation_history
                )
                
                assistant_message = response.content[0].text
                
                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                return assistant_message
                
            except Exception as e:
                logger.error(f"Chat error: {e}")
                return f"Error: {e}"
        else:
            return "Claude API not available. Please configure ANTHROPIC_API_KEY."
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []


class StrategyAssistant:
    """
    High-level assistant combining all strategy generation capabilities.
    """
    
    def __init__(self,
                 anthropic_api_key: Optional[str] = None,
                 vector_db=None,
                 ml_models=None):
        self.generator = F1StrategyGenerator(api_key=anthropic_api_key)
        self.vector_db = vector_db
        self.ml_models = ml_models
    
    def get_race_strategy(self, request: StrategyRequest) -> StrategyResponse:
        """Get a complete race strategy with all available context"""
        # Get historical context from vector DB
        historical_context = []
        if self.vector_db:
            try:
                query = f"Race strategy for {request.circuit} in {request.weather_forecast} conditions"
                historical_context = self.vector_db.search_similar_strategies(query, top_k=5)
            except Exception as e:
                logger.warning(f"Could not get historical context: {e}")
        
        # Get ML predictions
        ml_predictions = None
        if self.ml_models:
            try:
                # This would use the actual ML models
                pass
            except Exception as e:
                logger.warning(f"Could not get ML predictions: {e}")
        
        # Generate strategy
        race_info = {
            'circuit': request.circuit,
            'weather_forecast': request.weather_forecast,
            'total_laps': request.total_laps,
            'driver': request.driver,
            'constructor': request.constructor,
            'grid_position': request.grid_position,
            'tire_compounds': request.tire_compounds_available,
            'safety_car_probability': request.safety_car_probability
        }
        
        return self.generator.generate_race_strategy(
            race_info=race_info,
            historical_context=historical_context,
            ml_predictions=ml_predictions
        )
    
    def quick_recommendation(self, 
                             circuit: str,
                             current_lap: int,
                             total_laps: int,
                             current_tire: str,
                             tire_age: int,
                             position: int) -> Dict[str, Any]:
        """Get a quick pit stop recommendation"""
        prompt = f"""Quick F1 pit stop recommendation:

Current state:
- Circuit: {circuit}
- Lap {current_lap} of {total_laps}
- Position: P{position}
- Tire: {current_tire} ({tire_age} laps old)

Should we pit now? If yes, which compound?
Respond with JSON: {{"pit_now": boolean, "recommended_compound": string, "reasoning": string}}"""

        response = self.generator._call_claude(prompt, max_tokens=500)
        
        try:
            return self.generator._extract_json(response)
        except:
            return {
                "pit_now": tire_age > 25,
                "recommended_compound": "HARD" if tire_age > 25 else current_tire,
                "reasoning": response[:200]
            }


if __name__ == "__main__":
    # Test strategy generator
    print("Testing F1 Strategy Generator...")
    
    generator = F1StrategyGenerator()
    
    # Test strategy generation
    race_info = {
        'circuit': 'Monaco',
        'weather_forecast': 'Dry, 24°C, 10% chance of rain',
        'total_laps': 78,
        'driver': 'Max Verstappen',
        'constructor': 'Red Bull Racing',
        'grid_position': 1
    }
    
    print("\nGenerating race strategy...")
    strategy = generator.generate_race_strategy(race_info)
    
    print(f"\n📋 Executive Summary:")
    print(f"   {strategy.executive_summary[:200]}...")
    
    print(f"\n🎯 Recommended Strategy:")
    print(f"   {json.dumps(strategy.recommended_strategy, indent=2)[:300]}...")
    
    print(f"\n⚠️  Risk Assessment:")
    print(f"   {json.dumps(strategy.risk_assessment, indent=2)[:200]}...")
    
    print(f"\n🌧️  Weather Contingency:")
    print(f"   {strategy.weather_contingency[:200]}...")
    
    print(f"\n📊 Confidence Score: {strategy.confidence_score:.0%}")
    
    print("\n✅ Strategy generator test complete!")
