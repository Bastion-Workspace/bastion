"""
Historical Oregon Trail landmarks and constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Landmark:
    name: str
    mile_marker: int
    terrain: str
    has_fort: bool = False
    has_river: bool = False
    river_width_ft: Optional[int] = None
    river_base_depth_ft: Optional[float] = None
    description_hint: str = ""


TRAIL_LANDMARKS: List[Landmark] = [
    Landmark("Independence, Missouri", 0, "prairie",
             description_hint="Bustling frontier town, jumping-off point for the trail"),
    Landmark("Kansas River Crossing", 82, "prairie", has_river=True,
             river_width_ft=620, river_base_depth_ft=4.0,
             description_hint="Wide, muddy river with shifting sandbars"),
    Landmark("Big Blue River Crossing", 185, "prairie", has_river=True,
             river_width_ft=200, river_base_depth_ft=3.5,
             description_hint="Clear water but steep banks"),
    Landmark("Fort Kearney", 304, "prairie", has_fort=True,
             description_hint="First military post on the trail, a welcome sight"),
    Landmark("Chimney Rock", 554, "plains",
             description_hint="Towering spire visible for days, a milestone"),
    Landmark("Fort Laramie", 640, "plains", has_fort=True,
             description_hint="Major resupply point in Wyoming Territory"),
    Landmark("Independence Rock", 830, "hills",
             description_hint="Massive granite dome — emigrants carved their names here"),
    Landmark("South Pass", 932, "mountains",
             description_hint="Gentle crossing of the Continental Divide at 7,550 ft"),
    Landmark("Fort Bridger", 1000, "mountains", has_fort=True,
             description_hint="Trading post in the Green River valley"),
    Landmark("Soda Springs", 1160, "mountains",
             description_hint="Natural mineral springs that fizz like soda water"),
    Landmark("Fort Hall", 1210, "desert", has_fort=True,
             description_hint="Hudson's Bay Company post on the Snake River plain"),
    Landmark("Snake River Crossing", 1390, "desert", has_river=True,
             river_width_ft=1000, river_base_depth_ft=6.0,
             description_hint="Treacherous crossing — deep, swift current"),
    Landmark("Fort Boise", 1520, "desert", has_fort=True,
             description_hint="Last fort before the Blue Mountains"),
    Landmark("Blue Mountains", 1700, "mountains",
             description_hint="Steep, forested range — the final mountain barrier"),
    Landmark("The Dalles", 1930, "mountains", has_river=True,
             river_width_ft=1200, river_base_depth_ft=8.0,
             description_hint="Dangerous Columbia River rapids"),
    Landmark("Willamette Valley (Oregon City)", 2170, "valley",
             description_hint="Journey's end — fertile farmland and new beginnings"),
]

SHOP_PRICES = {
    "food": 0.20,
    "ammunition": 2.00,
    "clothing": 10.00,
    "spare_parts": 10.00,
    "oxen": 40.00,
}

FORT_PRICE_MULTIPLIERS = {
    "Fort Kearney": 1.0,
    "Fort Laramie": 1.4,
    "Fort Bridger": 1.6,
    "Fort Hall": 1.8,
    "Fort Boise": 2.0,
}

TERRAIN_SPEED_MODIFIER = {
    "prairie": 1.0,
    "plains": 0.95,
    "hills": 0.8,
    "mountains": 0.6,
    "desert": 0.75,
    "valley": 0.9,
}

WEATHER_SPEED_MODIFIER = {
    "clear": 1.0,
    "cloudy": 0.95,
    "rain": 0.7,
    "heavy_rain": 0.4,
    "snow": 0.3,
    "blizzard": 0.1,
    "hot": 0.85,
    "fog": 0.6,
}

PACE_BASE_MILES = {
    "steady": 14,
    "strenuous": 18,
    "grueling": 22,
}

PACE_HEALTH_PENALTY = {
    "steady": 0,
    "strenuous": 2,
    "grueling": 5,
}

RATIONS_FOOD_PER_PERSON = {
    "filling": 3,
    "meager": 2,
    "bare_bones": 1,
}

RATIONS_HEALTH_BONUS = {
    "filling": 2,
    "meager": 0,
    "bare_bones": -3,
}

MONTH_WEATHER_WEIGHTS = {
    4: {"clear": 35, "cloudy": 25, "rain": 25, "heavy_rain": 10, "fog": 5},
    5: {"clear": 40, "cloudy": 20, "rain": 25, "heavy_rain": 10, "fog": 5},
    6: {"clear": 45, "cloudy": 20, "rain": 15, "hot": 15, "fog": 5},
    7: {"clear": 35, "cloudy": 15, "rain": 10, "hot": 30, "fog": 5, "heavy_rain": 5},
    8: {"clear": 40, "cloudy": 20, "rain": 15, "hot": 20, "fog": 5},
    9: {"clear": 35, "cloudy": 25, "rain": 20, "heavy_rain": 10, "fog": 10},
    10: {"clear": 25, "cloudy": 25, "rain": 15, "snow": 20, "fog": 10, "heavy_rain": 5},
    11: {"clear": 15, "cloudy": 20, "snow": 30, "blizzard": 15, "fog": 10, "rain": 10},
}

AILMENTS = [
    "dysentery", "typhoid", "cholera", "measles", "snakebite",
    "broken_arm", "broken_leg", "exhaustion", "fever",
]

AILMENT_DAILY_DAMAGE = {
    "dysentery": 5,
    "typhoid": 8,
    "cholera": 12,
    "measles": 4,
    "snakebite": 6,
    "broken_arm": 2,
    "broken_leg": 3,
    "exhaustion": 3,
    "fever": 4,
}

AILMENT_RECOVERY_CHANCE = {
    "dysentery": 0.15,
    "typhoid": 0.08,
    "cholera": 0.05,
    "measles": 0.20,
    "snakebite": 0.12,
    "broken_arm": 0.10,
    "broken_leg": 0.07,
    "exhaustion": 0.30,
    "fever": 0.18,
}
