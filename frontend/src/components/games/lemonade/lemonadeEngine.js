const WEATHER_TYPES = ['scorching', 'sunny', 'cloudy', 'overcast', 'rainy'];
const WEATHER_LABELS = { scorching: 'Scorching', sunny: 'Sunny', cloudy: 'Cloudy', overcast: 'Overcast', rainy: 'Rainy' };
const WEATHER_DEMAND_MULT = { scorching: 1.8, sunny: 1.4, cloudy: 1.0, overcast: 0.6, rainy: 0.3 };
const WEATHER_PAY_MULT = { scorching: 1.2, sunny: 1.0, cloudy: 0.9, overcast: 0.8, rainy: 0.6 };

const UPGRADES = [
  { id: 'cups', name: 'Better Cups', cost: 5, effect: 'customers', mult: 1.05 },
  { id: 'sign', name: 'Roadside Sign', cost: 15, effect: 'customers', mult: 1.15 },
  { id: 'ice', name: 'Premium Ice', cost: 10, effect: 'pay', mult: 1.1 },
  { id: 'bulk', name: 'Bulk Lemons', cost: 20, effect: 'cost', mult: 0.85 },
];

const LEMON_COST = 0.05;
const CUP_COST = 0.02;
const ICE_COST = 0.01;
const BASE_CUSTOMERS = 80;
const PRICE_ELASTICITY = 2.5;
const STARTING_MONEY = 5;
const TOTAL_DAYS = 30;

let rng = () => Math.random();
export function setSeed(seed) {
  if (seed !== undefined) rng = () => (Math.sin(seed++) * 10000) % 1;
}

function pickWeather(day) {
  const roll = rng();
  if (roll < 0.1) return 'scorching';
  if (roll < 0.35) return 'sunny';
  if (roll < 0.6) return 'cloudy';
  if (roll < 0.85) return 'overcast';
  return 'rainy';
}

export function getForecast(day) {
  const actual = pickWeather(day);
  const inaccurate = rng() < 0.2;
  if (inaccurate) {
    const idx = WEATHER_TYPES.indexOf(actual);
    const offset = rng() < 0.5 ? -1 : 1;
    const wrong = WEATHER_TYPES[Math.max(0, Math.min(WEATHER_TYPES.length - 1, idx + offset))];
    return { forecast: wrong, actual };
  }
  return { forecast: actual, actual };
}

function getEvent(day) {
  const roll = rng();
  if (roll < 0.08) return { type: 'newspaper', message: 'A newspaper article featured your stand! Customers flock in.' };
  if (roll < 0.14) return { type: 'heatwave', message: 'A heat wave doubled thirst for lemonade.' };
  if (roll < 0.19) return { type: 'rain_spike', message: 'Unexpected downpour reduced foot traffic.' };
  if (roll < 0.22) return { type: 'contest', message: 'A local contest gave you a small prize. +$2.' };
  return null;
}

export function newGame() {
  const day1Weather = getForecast(1);
  return {
    day: 1,
    money: STARTING_MONEY,
    lemons: 0,
    cups: 0,
    ice: 0,
    upgrades: [],
    history: [],
    phase: 'planning',
    forecast: day1Weather.forecast,
    event: null,
  };
}

export function runDay(state, choices) {
  const { pricePerCup, buyLemons, buyCups, buyIce, buyUpgradeId } = choices;
  const weatherResult = getForecast(state.day);
  const actualWeather = weatherResult.actual;
  const event = getEvent(state.day);

  let money = state.money;
  let lemons = state.lemons + (buyLemons || 0);
  let cups = state.cups + (buyCups || 0);
  let ice = state.ice + (buyIce || 0);

  const costLemons = (buyLemons || 0) * LEMON_COST;
  const costCups = (buyCups || 0) * CUP_COST;
  const costIce = (buyIce || 0) * ICE_COST;
  let costUpgrade = 0;
  const upgrades = [...state.upgrades];
  if (buyUpgradeId && !upgrades.includes(buyUpgradeId)) {
    const up = UPGRADES.find((u) => u.id === buyUpgradeId);
    if (up && money >= up.cost) {
      costUpgrade = up.cost;
      upgrades.push(buyUpgradeId);
    }
  }

  money -= costLemons + costCups + costIce + costUpgrade;
  if (event?.type === 'contest') money += 2;

  let customerMult = WEATHER_DEMAND_MULT[actualWeather];
  let payMult = WEATHER_PAY_MULT[actualWeather];
  for (const uid of upgrades) {
    const u = UPGRADES.find((x) => x.id === uid);
    if (!u) continue;
    if (u.effect === 'customers') customerMult *= u.mult;
    if (u.effect === 'pay') payMult *= u.mult;
  }
  if (event?.type === 'newspaper') customerMult *= 1.5;
  if (event?.type === 'heatwave') customerMult *= 2;
  if (event?.type === 'rain_spike') customerMult *= 0.5;

  const price = Math.max(0.01, Number(pricePerCup) || 0.25);
  const demandFactor = 1 / (1 + Math.pow(price, PRICE_ELASTICITY));
  const potentialCustomers = Math.max(0, Math.floor(BASE_CUSTOMERS * customerMult * demandFactor));
  const maxCups = Math.min(lemons * 4, cups, ice > 0 ? Math.min(lemons * 4, cups, ice * 2) : Math.min(lemons * 4, cups));
  const customers = Math.min(potentialCustomers, maxCups);
  const effectivePrice = price * payMult;
  let costPerCup = LEMON_COST * 0.25 + CUP_COST + (ice > 0 ? ICE_COST * 0.5 : 0);
  for (const uid of upgrades) {
    const u = UPGRADES.find((x) => x.id === uid);
    if (u?.effect === 'cost') costPerCup *= u.mult;
  }
  const revenue = customers * effectivePrice;
  const costOfGoods = customers * costPerCup;
  const profit = revenue - costOfGoods;
  money += profit;

  const usedLemons = Math.ceil(customers / 4);
  const usedCups = customers;
  const usedIce = Math.min(ice, Math.ceil(customers / 2));
  lemons -= usedLemons;
  cups -= usedCups;
  ice -= usedIce;

  const nextDay = state.day + 1;
  const historyEntry = {
    day: state.day,
    weather: actualWeather,
    customers,
    revenue,
    cost: costOfGoods,
    profit,
    event: event?.type,
  };

  const nextState = {
    ...state,
    day: nextDay,
    money: Math.round(money * 100) / 100,
    lemons,
    cups,
    ice,
    upgrades,
    history: [...state.history, historyEntry],
    phase: nextDay > TOTAL_DAYS ? 'gameover' : 'results',
    lastResult: {
      ...historyEntry,
      eventMessage: event?.message,
      event,
    },
    forecast: nextDay <= TOTAL_DAYS ? getForecast(nextDay).forecast : state.forecast,
  };
  return nextState;
}

export function getNextForecast(state) {
  return getForecast(state.day + 1);
}

export { WEATHER_TYPES, WEATHER_LABELS, UPGRADES, LEMON_COST, CUP_COST, ICE_COST, TOTAL_DAYS, STARTING_MONEY };
