"""
Weather Response Formatters
Handles all weather response formatting with user preference adaptation
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class WeatherResponseFormatters:
    """Handles all weather response formatting with multiple communication styles"""
    
    def __init__(self):
        pass
    
    async def format_weather_response(
        self, 
        request: Dict[str, Any], 
        weather_data: Dict[str, Any], 
        communication_style: str,
        detail_level: str,
        shared_memory: Dict[str, Any]
    ) -> str:
        """Format weather response based on user communication preferences"""
        try:
            if request["request_type"] == "current":
                return await self.format_current_conditions_response(
                    weather_data, communication_style, detail_level, shared_memory
                )
            elif request["request_type"] == "history":
                return await self.format_historical_response(
                    weather_data, communication_style, detail_level, shared_memory, request.get("date_str", "")
                )
            else:
                return await self.format_forecast_response(
                    weather_data, communication_style, detail_level, shared_memory, request["forecast_days"]
                )
                
        except Exception as e:
            logger.error(f"❌ Error formatting weather response: {e}")
            return f"I have the weather data, but encountered an error formatting it: {str(e)}"
    
    async def format_current_conditions_response(
        self, 
        weather_data: Dict[str, Any], 
        style: str, 
        detail_level: str,
        shared_memory: Dict[str, Any]
    ) -> str:
        """Format current weather conditions response"""
        try:
            current = weather_data["current"]
            location = weather_data["location"]
            units = weather_data["units"]
            
            temp = current["temperature"]
            feels_like = current["feels_like"]
            conditions = current["conditions"]
            humidity = current["humidity"]
            wind_speed = current["wind_speed"]
            
            # Get historical context from shared memory
            historical_context = self._get_historical_context(location["query"], shared_memory)
            
            # Base response components
            temp_info = f"{temp:.0f}{units['temperature']}"
            if abs(temp - feels_like) > 3:
                temp_info += f" (feels like {feels_like:.0f}{units['temperature']})"
            
            location_info = f"{location['name']}"
            if location.get("country") and location["country"] != "US":
                location_info += f", {location['country']}"
            
            # Adapt response to communication style
            if style == "enthusiastic" or style == "roosevelt":
                response = f"**BULLY!** The weather in {location_info} is absolutely splendid! "
                response += f"We've got {temp_info} with {conditions.lower()}. "
                
                if detail_level in ["detailed", "high"]:
                    response += f"**By George!** The humidity is at {humidity}%, "
                    response += f"and the wind is charging along at {wind_speed:.1f} {units['wind_speed']}. "
                    
                    if current.get("visibility"):
                        response += f"Visibility is a magnificent {current['visibility']:.1f} {units['visibility']}! "
                
                # Add weather advice in Roosevelt style
                response += self._get_roosevelt_weather_advice(current, style)
                
            elif style == "professional" or style == "formal":
                response = f"Current weather conditions in {location_info}: "
                response += f"{temp_info}, {conditions.lower()}. "
                
                if detail_level in ["detailed", "high"]:
                    response += f"Humidity: {humidity}%, wind: {wind_speed:.1f} {units['wind_speed']}. "
                    if current.get("visibility"):
                        response += f"Visibility: {current['visibility']:.1f} {units['visibility']}. "
                
                response += self._get_professional_weather_advice(current)
                
            elif style == "casual" or style == "friendly":
                response = f"Hey! It's {temp_info} in {location_info} right now with {conditions.lower()}. "
                
                if detail_level in ["detailed", "high"]:
                    response += f"Pretty {humidity}% humid, and there's a {wind_speed:.1f} {units['wind_speed']} breeze. "
                
                response += self._get_casual_weather_advice(current)
                
            else:  # Default neutral style
                response = f"The current weather in {location_info} is {temp_info} with {conditions.lower()}. "
                
                if detail_level in ["detailed", "high"]:
                    response += f"Humidity is {humidity}% with wind at {wind_speed:.1f} {units['wind_speed']}. "
                
                response += self._get_neutral_weather_advice(current)
            
            # Add historical context if available
            if historical_context:
                response += f"\n\n{historical_context}"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error formatting current conditions: {e}")
            return f"I have the current weather data but encountered a formatting error: {str(e)}"
    
    async def format_forecast_response(
        self, 
        weather_data: Dict[str, Any], 
        style: str, 
        detail_level: str,
        shared_memory: Dict[str, Any],
        days: int
    ) -> str:
        """Format weather forecast response"""
        try:
            forecast = weather_data["forecast"]
            location = weather_data["location"]
            units = weather_data["units"]
            
            location_info = f"{location['name']}"
            if location.get("country") and location["country"] != "US":
                location_info += f", {location['country']}"
            
            # Adapt response to communication style
            if style == "enthusiastic" or style == "roosevelt":
                response = f"**BULLY!** Here's the magnificent {days}-day weather forecast for {location_info}:\n\n"
                
                for day in forecast:
                    day_name = day["day_name"]
                    high = day["temperature"]["high"]
                    low = day["temperature"]["low"]
                    conditions = day["conditions"]
                    precip_prob = day["precipitation_probability"]
                    
                    response += f"**{day_name}**: A splendid {high:.0f}{units['temperature']} high and {low:.0f}{units['temperature']} low with {conditions.lower()}. "
                    
                    if precip_prob > 30:
                        response += f"**By George!** There's a {precip_prob:.0f}% chance of precipitation - perfect for a cavalry charge with an umbrella! "
                    
                    if detail_level in ["detailed", "high"]:
                        humidity = day["humidity"]
                        wind = day["wind_speed"]
                        response += f"(Humidity: {humidity:.0f}%, Wind: {wind:.1f} {units['wind_speed']}) "
                    
                    response += "\n"
                
                response += "\n**Trust busting weather for outdoor activities!**"
                
            elif style == "professional" or style == "formal":
                response = f"{days}-day weather forecast for {location_info}:\n\n"
                
                for day in forecast:
                    day_name = day["day_name"]
                    high = day["temperature"]["high"]
                    low = day["temperature"]["low"]
                    conditions = day["conditions"]
                    precip_prob = day["precipitation_probability"]
                    
                    response += f"{day_name}: High {high:.0f}{units['temperature']}, Low {low:.0f}{units['temperature']}, {conditions}. "
                    
                    if precip_prob > 20:
                        response += f"Precipitation probability: {precip_prob:.0f}%. "
                    
                    if detail_level in ["detailed", "high"]:
                        humidity = day["humidity"]
                        wind = day["wind_speed"]
                        response += f"Humidity: {humidity:.0f}%, Wind: {wind:.1f} {units['wind_speed']}. "
                    
                    response += "\n"
                
            elif style == "casual" or style == "friendly":
                response = f"Here's what the weather looks like for the next {days} days in {location_info}:\n\n"
                
                for day in forecast:
                    day_name = day["day_name"]
                    high = day["temperature"]["high"]
                    low = day["temperature"]["low"]
                    conditions = day["conditions"]
                    precip_prob = day["precipitation_probability"]
                    
                    response += f"{day_name}: {high:.0f}{units['temperature']}/{low:.0f}{units['temperature']} with {conditions.lower()}. "
                    
                    if precip_prob > 40:
                        response += f"Might rain ({precip_prob:.0f}% chance) - grab an umbrella! "
                    
                    response += "\n"
                
                response += "\nHope that helps with your plans! 😊"
                
            else:  # Default neutral style
                response = f"Weather forecast for {location_info} ({days} days):\n\n"
                
                for day in forecast:
                    day_name = day["day_name"]
                    high = day["temperature"]["high"]
                    low = day["temperature"]["low"]
                    conditions = day["conditions"]
                    precip_prob = day["precipitation_probability"]
                    
                    response += f"{day_name}: {high:.0f}{units['temperature']}/{low:.0f}{units['temperature']}, {conditions}. "
                    
                    if precip_prob > 30:
                        response += f"Precipitation: {precip_prob:.0f}%. "
                    
                    response += "\n"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error formatting forecast: {e}")
            return f"I have the forecast data but encountered a formatting error: {str(e)}"
    
    async def format_historical_response(
        self,
        weather_data: Dict[str, Any],
        style: str,
        detail_level: str,
        shared_memory: Dict[str, Any],
        date_str: str
    ) -> str:
        """Format historical weather response"""
        try:
            historical = weather_data.get("historical", {})
            location = weather_data.get("location", {})
            period = weather_data.get("period", {})
            units = weather_data.get("units", {})
            
            location_info = location.get("name", "Unknown")
            if location.get("country") and location["country"] != "US":
                location_info += f", {location['country']}"
            
            period_type = period.get("type", "daily")
            is_monthly = period_type == "monthly_average"
            is_yearly = period_type == "yearly_summary"
            
            # Adapt response to communication style
            if style == "enthusiastic" or style == "roosevelt":
                if is_yearly:
                    # Yearly summary with monthly breakdown
                    year = period.get("year", date_str)
                    yearly_avg = historical.get("yearly_average_temperature", 0)
                    yearly_min = historical.get("yearly_min_temperature", 0)
                    yearly_max = historical.get("yearly_max_temperature", 0)
                    monthly_data = historical.get("monthly_data", [])
                    temp_unit = units.get("temperature", "°F")
                    
                    response = f"**BULLY!** Historical weather intelligence for {location_info} in {year}:\n\n"
                    response += f"**Yearly Average Temperature**: {yearly_avg:.1f}{temp_unit}\n"
                    response += f"**Yearly Temperature Range**: {yearly_min:.1f}{temp_unit} to {yearly_max:.1f}{temp_unit}\n\n"
                    response += f"**Monthly Breakdown**:\n"
                    
                    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    for month_data in monthly_data:
                        month_num = month_data.get("month", 0)
                        if 1 <= month_num <= 12:
                            month_name = month_names[month_num - 1]
                            avg_temp = month_data.get("average_temperature", 0)
                            response += f"- **{month_name}**: {avg_temp:.1f}{temp_unit}\n"
                    
                    response += "\n**By George!** Complete meteorological intelligence for the entire year!"
                elif is_monthly:
                    avg_temp = historical.get("average_temperature", 0)
                    min_temp = historical.get("min_temperature", 0)
                    max_temp = historical.get("max_temperature", 0)
                    temp_unit = units.get("temperature", "°F")
                    month_name = datetime.strptime(date_str, "%Y-%m").strftime("%B %Y")
                    
                    response = f"**BULLY!** Historical weather intelligence for {location_info} in {month_name}:\n\n"
                    response += f"**Average Temperature**: {avg_temp:.1f}{temp_unit}\n"
                    response += f"**Temperature Range**: {min_temp:.1f}{temp_unit} to {max_temp:.1f}{temp_unit}\n"
                    
                    if detail_level in ["detailed", "high"]:
                        avg_humidity = historical.get("average_humidity", 0)
                        avg_wind = historical.get("average_wind_speed", 0)
                        wind_unit = units.get("wind_speed", "mph")
                        conditions = historical.get("most_common_conditions", "N/A")
                        response += f"**Average Humidity**: {avg_humidity:.1f}%\n"
                        response += f"**Average Wind Speed**: {avg_wind:.1f} {wind_unit}\n"
                        response += f"**Most Common Conditions**: {conditions}\n"
                        response += f"**Sample Days**: {historical.get('sample_days', 0)} days sampled\n"
                    
                    response += "\n**By George!** This meteorological intelligence is based on historical data!"
                else:
                    temp = historical.get("temperature", 0)
                    feels_like = historical.get("feels_like", 0)
                    conditions = historical.get("conditions", "")
                    temp_unit = units.get("temperature", "°F")
                    date_display = period.get("date", date_str)
                    
                    response = f"**BULLY!** Historical weather for {location_info} on {date_display}:\n\n"
                    response += f"**Temperature**: {temp:.1f}{temp_unit}"
                    if abs(temp - feels_like) > 3:
                        response += f" (felt like {feels_like:.1f}{temp_unit})"
                    response += f"\n**Conditions**: {conditions}\n"
                    
                    if detail_level in ["detailed", "high"]:
                        humidity = historical.get("humidity", 0)
                        wind = historical.get("wind_speed", 0)
                        wind_unit = units.get("wind_speed", "mph")
                        pressure = historical.get("pressure", 0)
                        response += f"**Humidity**: {humidity}%\n"
                        response += f"**Wind Speed**: {wind:.1f} {wind_unit}\n"
                        response += f"**Pressure**: {pressure} hPa\n"
                    
                    response += "\n**Trust busting historical data!**"
                    
            elif style == "professional" or style == "formal":
                if is_yearly:
                    # Yearly summary with monthly breakdown
                    year = period.get("year", date_str)
                    yearly_avg = historical.get("yearly_average_temperature", 0)
                    yearly_min = historical.get("yearly_min_temperature", 0)
                    yearly_max = historical.get("yearly_max_temperature", 0)
                    monthly_data = historical.get("monthly_data", [])
                    temp_unit = units.get("temperature", "°F")
                    
                    response = f"Historical weather data for {location_info}, {year}:\n\n"
                    response += f"Yearly Average Temperature: {yearly_avg:.1f}{temp_unit}\n"
                    response += f"Yearly Temperature Range: {yearly_min:.1f}{temp_unit} to {yearly_max:.1f}{temp_unit}\n\n"
                    response += f"Monthly Averages:\n"
                    
                    month_names = ["January", "February", "March", "April", "May", "June", 
                                  "July", "August", "September", "October", "November", "December"]
                    for month_data in monthly_data:
                        month_num = month_data.get("month", 0)
                        if 1 <= month_num <= 12:
                            month_name = month_names[month_num - 1]
                            avg_temp = month_data.get("average_temperature", 0)
                            response += f"- {month_name}: {avg_temp:.1f}{temp_unit}\n"
                elif is_monthly:
                    avg_temp = historical.get("average_temperature", 0)
                    min_temp = historical.get("min_temperature", 0)
                    max_temp = historical.get("max_temperature", 0)
                    temp_unit = units.get("temperature", "°F")
                    month_name = datetime.strptime(date_str, "%Y-%m").strftime("%B %Y")
                    
                    response = f"Historical weather data for {location_info}, {month_name}:\n\n"
                    response += f"Average Temperature: {avg_temp:.1f}{temp_unit}\n"
                    response += f"Temperature Range: {min_temp:.1f}{temp_unit} to {max_temp:.1f}{temp_unit}\n"
                    
                    if detail_level in ["detailed", "high"]:
                        avg_humidity = historical.get("average_humidity", 0)
                        avg_wind = historical.get("average_wind_speed", 0)
                        wind_unit = units.get("wind_speed", "mph")
                        conditions = historical.get("most_common_conditions", "N/A")
                        response += f"Average Humidity: {avg_humidity:.1f}%\n"
                        response += f"Average Wind Speed: {avg_wind:.1f} {wind_unit}\n"
                        response += f"Most Common Conditions: {conditions}\n"
                        response += f"Data Points: {historical.get('sample_days', 0)} days sampled\n"
                else:
                    temp = historical.get("temperature", 0)
                    conditions = historical.get("conditions", "")
                    temp_unit = units.get("temperature", "°F")
                    date_display = period.get("date", date_str)
                    
                    response = f"Historical weather for {location_info} on {date_display}:\n\n"
                    response += f"Temperature: {temp:.1f}{temp_unit}\n"
                    response += f"Conditions: {conditions}\n"
                    
                    if detail_level in ["detailed", "high"]:
                        humidity = historical.get("humidity", 0)
                        wind = historical.get("wind_speed", 0)
                        wind_unit = units.get("wind_speed", "mph")
                        pressure = historical.get("pressure", 0)
                        response += f"Humidity: {humidity}%\n"
                        response += f"Wind Speed: {wind:.1f} {wind_unit}\n"
                        response += f"Pressure: {pressure} hPa\n"
                        
            elif style == "casual" or style == "friendly":
                if is_yearly:
                    # Yearly summary with monthly breakdown
                    year = period.get("year", date_str)
                    yearly_avg = historical.get("yearly_average_temperature", 0)
                    monthly_data = historical.get("monthly_data", [])
                    temp_unit = units.get("temperature", "°F")
                    
                    response = f"Here's the historical weather for {location_info} in {year}:\n\n"
                    response += f"Yearly average: {yearly_avg:.1f}{temp_unit}\n\n"
                    response += f"Monthly breakdown:\n"
                    
                    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    for month_data in monthly_data:
                        month_num = month_data.get("month", 0)
                        if 1 <= month_num <= 12:
                            month_name = month_names[month_num - 1]
                            avg_temp = month_data.get("average_temperature", 0)
                            response += f"- {month_name}: {avg_temp:.1f}{temp_unit}\n"
                    
                    response += "\nHope that helps! 😊"
                elif is_monthly:
                    avg_temp = historical.get("average_temperature", 0)
                    temp_unit = units.get("temperature", "°F")
                    month_name = datetime.strptime(date_str, "%Y-%m").strftime("%B %Y")
                    
                    response = f"Here's the historical weather for {location_info} in {month_name}:\n\n"
                    response += f"Average temp: {avg_temp:.1f}{temp_unit}\n"
                    
                    if detail_level in ["detailed", "high"]:
                        min_temp = historical.get("min_temperature", 0)
                        max_temp = historical.get("max_temperature", 0)
                        response += f"Range: {min_temp:.1f}{temp_unit} to {max_temp:.1f}{temp_unit}\n"
                        conditions = historical.get("most_common_conditions", "N/A")
                        response += f"Mostly: {conditions}\n"
                else:
                    temp = historical.get("temperature", 0)
                    conditions = historical.get("conditions", "")
                    temp_unit = units.get("temperature", "°F")
                    date_display = period.get("date", date_str)
                    
                    response = f"Historical weather for {location_info} on {date_display}:\n\n"
                    response += f"Temp: {temp:.1f}{temp_unit}, {conditions}\n"
                    
                    if detail_level in ["detailed", "high"]:
                        humidity = historical.get("humidity", 0)
                        wind = historical.get("wind_speed", 0)
                        response += f"Humidity: {humidity}%, Wind: {wind:.1f} {units.get('wind_speed', 'mph')}\n"
                    
                    response += "\nHope that helps! 😊"
                    
            else:  # Default neutral style
                if is_yearly:
                    # Yearly summary with monthly breakdown
                    year = period.get("year", date_str)
                    yearly_avg = historical.get("yearly_average_temperature", 0)
                    yearly_min = historical.get("yearly_min_temperature", 0)
                    yearly_max = historical.get("yearly_max_temperature", 0)
                    monthly_data = historical.get("monthly_data", [])
                    temp_unit = units.get("temperature", "°F")
                    
                    response = f"Historical weather for {location_info}, {year}:\n\n"
                    response += f"Yearly Average: {yearly_avg:.1f}{temp_unit} (Range: {yearly_min:.1f}{temp_unit} to {yearly_max:.1f}{temp_unit})\n\n"
                    response += f"Monthly Averages:\n"
                    
                    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    for month_data in monthly_data:
                        month_num = month_data.get("month", 0)
                        if 1 <= month_num <= 12:
                            month_name = month_names[month_num - 1]
                            avg_temp = month_data.get("average_temperature", 0)
                            response += f"- {month_name}: {avg_temp:.1f}{temp_unit}\n"
                elif is_monthly:
                    avg_temp = historical.get("average_temperature", 0)
                    min_temp = historical.get("min_temperature", 0)
                    max_temp = historical.get("max_temperature", 0)
                    temp_unit = units.get("temperature", "°F")
                    month_name = datetime.strptime(date_str, "%Y-%m").strftime("%B %Y")
                    
                    response = f"Historical weather for {location_info}, {month_name}:\n\n"
                    response += f"Average: {avg_temp:.1f}{temp_unit} (Range: {min_temp:.1f}{temp_unit} to {max_temp:.1f}{temp_unit})\n"
                    
                    if detail_level in ["detailed", "high"]:
                        avg_humidity = historical.get("average_humidity", 0)
                        avg_wind = historical.get("average_wind_speed", 0)
                        conditions = historical.get("most_common_conditions", "N/A")
                        response += f"Humidity: {avg_humidity:.1f}%, Wind: {avg_wind:.1f} {units.get('wind_speed', 'mph')}\n"
                        response += f"Conditions: {conditions}\n"
                else:
                    temp = historical.get("temperature", 0)
                    conditions = historical.get("conditions", "")
                    temp_unit = units.get("temperature", "°F")
                    date_display = period.get("date", date_str)
                    
                    response = f"Historical weather for {location_info} on {date_display}:\n\n"
                    response += f"Temperature: {temp:.1f}{temp_unit}, {conditions}\n"
                    
                    if detail_level in ["detailed", "high"]:
                        humidity = historical.get("humidity", 0)
                        wind = historical.get("wind_speed", 0)
                        response += f"Humidity: {humidity}%, Wind: {wind:.1f} {units.get('wind_speed', 'mph')}\n"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error formatting historical weather: {e}")
            return f"I have the historical weather data but encountered a formatting error: {str(e)}"
    
    def _get_historical_context(self, location: str, shared_memory: Dict[str, Any]) -> Optional[str]:
        """Get historical weather context from shared memory"""
        try:
            weather_data = shared_memory.get("weather_data", {})
            recent_queries = weather_data.get("recent_queries", {})
            
            if location in recent_queries:
                last_query = recent_queries[location]
                last_timestamp = last_query.get("timestamp", "")
                
                if last_timestamp:
                    last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                    time_diff = datetime.utcnow() - last_time.replace(tzinfo=None)
                    
                    if time_diff.days == 0 and time_diff.seconds < 3600:  # Less than 1 hour ago
                        return "📊 This is updated information from your recent weather query."
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting historical context: {e}")
            return None
    
    def _get_roosevelt_weather_advice(self, current: Dict[str, Any], style: str) -> str:
        """Get weather advice in Roosevelt style"""
        temp = current["temperature"]
        conditions = current["conditions"].lower()
        wind_speed = current["wind_speed"]
        
        advice = ""
        
        if "rain" in conditions:
            advice = "**Splendid weather for indoor strategic planning!** Grab your umbrella if you must venture forth!"
        elif "snow" in conditions:
            advice = "**By George!** Perfect weather for a vigorous winter campaign! Bundle up like a proper Rough Rider!"
        elif temp > 80:
            advice = "**BULLY!** Magnificent weather for outdoor adventures! Stay hydrated on your cavalry charges!"
        elif temp < 32:
            advice = "**Trust busting cold!** Perfect for building character - dress warmly for your battles!"
        elif wind_speed > 20:
            advice = "**By George!** Quite a wind out there - perfect for flying flags high!"
        else:
            advice = "**Absolutely magnificent weather** for any outdoor campaigns you might have planned!"
        
        return advice
    
    def _get_professional_weather_advice(self, current: Dict[str, Any]) -> str:
        """Get professional weather advice"""
        temp = current["temperature"]
        conditions = current["conditions"].lower()
        humidity = current["humidity"]
        
        if "rain" in conditions:
            return "Recommended to carry rain protection for outdoor activities."
        elif temp > 85 and humidity > 70:
            return "High temperature and humidity - consider limiting outdoor exposure during peak hours."
        elif temp < 32:
            return "Freezing conditions - appropriate winter clothing recommended."
        else:
            return "Favorable conditions for outdoor activities."
    
    def _get_casual_weather_advice(self, current: Dict[str, Any]) -> str:
        """Get casual weather advice"""
        temp = current["temperature"]
        conditions = current["conditions"].lower()
        
        if "rain" in conditions:
            return "Might want to grab an umbrella! ☔"
        elif temp > 80:
            return "Perfect weather to get outside! Don't forget sunscreen! ☀️"
        elif temp < 40:
            return "Brr! Might want to bundle up if you're heading out! 🧥"
        else:
            return "Pretty nice out there! 👍"
    
    def _get_neutral_weather_advice(self, current: Dict[str, Any]) -> str:
        """Get neutral weather advice"""
        temp = current["temperature"]
        conditions = current["conditions"].lower()
        
        if "rain" in conditions:
            return "Consider bringing rain protection."
        elif temp > 85:
            return "Warm conditions - stay hydrated if spending time outdoors."
        elif temp < 35:
            return "Cold conditions - dress appropriately for the temperature."
        else:
            return "Suitable conditions for most outdoor activities."
    
    async def format_error_response(self, error_message: str, style: str) -> str:
        """Format error response based on communication style"""
        if style == "enthusiastic" or style == "roosevelt":
            return f"**By George!** I encountered a challenge in my meteorological reconnaissance: {error_message}. **BULLY!** Let me know if you'd like to try again with a different location!"
        elif style == "professional" or style == "formal":
            return f"I apologize, but I encountered an error while retrieving weather information: {error_message}. Please verify the location and try again."
        elif style == "casual" or style == "friendly":
            return f"Oops! I ran into a problem getting that weather info: {error_message}. Want to try with a different location? 😅"
        else:
            return f"Error retrieving weather information: {error_message}. Please check the location and try again."


# Global instance
_formatter_instance = None


def get_weather_response_formatters() -> WeatherResponseFormatters:
    """Get global weather response formatters instance"""
    global _formatter_instance
    if _formatter_instance is None:
        _formatter_instance = WeatherResponseFormatters()
    return _formatter_instance

