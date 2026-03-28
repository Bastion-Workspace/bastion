const SUITS = ['s', 'h', 'd', 'c'];
const RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
const RANK_ORDER = { A: 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, J: 11, Q: 12, K: 13 };
const RED_SUITS = ['h', 'd'];
const BLACK_SUITS = ['s', 'c'];

function isRed(suit) {
  return RED_SUITS.includes(suit);
}

function isBlack(suit) {
  return BLACK_SUITS.includes(suit);
}

function oppositeColor(suitA, suitB) {
  return isRed(suitA) !== isRed(suitB);
}

export function createDeck() {
  const deck = [];
  for (const suit of SUITS) {
    for (const rank of RANKS) {
    deck.push({ suit, rank, faceUp: false, id: `${suit}-${rank}` });
    }
  }
  return deck;
}

export function shuffle(deck) {
  const arr = [...deck];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function newGame() {
  let deck = shuffle(createDeck());
  const tableau = [[], [], [], [], [], [], []];
  for (let col = 0; col < 7; col++) {
    for (let row = 0; row <= col; row++) {
      const card = deck.pop();
      card.faceUp = row === col;
      tableau[col].push(card);
    }
  }
  return {
    stock: deck,
    waste: [],
    foundations: [[], [], [], []],
    tableau,
    moves: 0,
    startTime: Date.now(),
    won: false,
  };
}

export function canMoveToFoundation(card, foundation) {
  if (foundation.length === 0) return card.rank === 'A';
  const top = foundation[foundation.length - 1];
  if (top.suit !== card.suit) return false;
  return RANK_ORDER[card.rank] === RANK_ORDER[top.rank] + 1;
}

export function canStackOnTableau(card, targetCard) {
  if (!targetCard) return card.rank === 'K';
  return oppositeColor(card.suit, targetCard.suit) && RANK_ORDER[card.rank] === RANK_ORDER[targetCard.rank] - 1;
}

export function canMoveFromTableau(tableau, tableauCol, cardIndex) {
  const col = tableau[tableauCol];
  if (!col || cardIndex < 0 || cardIndex >= col.length) return false;
  for (let i = cardIndex; i < col.length; i++) {
    if (!col[i].faceUp) return false;
    if (i > cardIndex && (RANK_ORDER[col[i].rank] !== RANK_ORDER[col[i - 1].rank] - 1 || !oppositeColor(col[i].suit, col[i - 1].suit))) return false;
  }
  return true;
}

export function findTableauDropColumn(card, tableau) {
  for (let c = 0; c < 7; c++) {
    const col = tableau[c];
    const top = col.length ? col[col.length - 1] : null;
    if (canStackOnTableau(card, top)) return c;
  }
  return -1;
}

export function findFoundationDropIndex(card, foundations) {
  for (let f = 0; f < 4; f++) {
    if (canMoveToFoundation(card, foundations[f])) return f;
  }
  return -1;
}

export function drawFromStock(state) {
  if (state.stock.length === 0 && state.waste.length === 0) return state;
  const next = { ...state, tableau: [...state.tableau], foundations: state.foundations.map((f) => [...f]) };
  if (next.stock.length === 0) {
    next.stock = next.waste.reverse();
    next.stock.forEach((c) => { c.faceUp = false; });
    next.waste = [];
  } else {
    const count = Math.min(3, next.stock.length);
    const drawn = next.stock.splice(-count);
    drawn.forEach((c) => { c.faceUp = true; });
    next.waste = [...next.waste, ...drawn];
  }
  next.moves = (next.moves || 0) + 1;
  return next;
}

export function moveTableauToTableau(state, fromCol, cardIndex, toCol) {
  if (fromCol === toCol) return state;
  const col = state.tableau[fromCol];
  if (!canMoveFromTableau(state.tableau, fromCol, cardIndex)) return state;
  const moving = col.slice(cardIndex);
  const toColArr = state.tableau[toCol];
  const top = toColArr.length ? toColArr[toColArr.length - 1] : null;
  if (!canStackOnTableau(moving[0], top)) return state;

  const next = JSON.parse(JSON.stringify(state));
  next.tableau[fromCol] = next.tableau[fromCol].slice(0, cardIndex);
  next.tableau[toCol] = [...next.tableau[toCol], ...moving];
  if (next.tableau[fromCol].length > 0) {
    const newTop = next.tableau[fromCol][next.tableau[fromCol].length - 1];
    if (!newTop.faceUp) newTop.faceUp = true;
  }
  next.moves = (next.moves || 0) + 1;
  return next;
}

export function moveTableauToFoundation(state, colIndex) {
  const col = state.tableau[colIndex];
  if (col.length === 0) return state;
  const card = col[col.length - 1];
  const fIndex = findFoundationDropIndex(card, state.foundations);
  if (fIndex < 0) return state;

  const next = JSON.parse(JSON.stringify(state));
  next.tableau[colIndex] = next.tableau[colIndex].slice(0, -1);
  next.foundations[fIndex].push(card);
  if (next.tableau[colIndex].length > 0) {
    const newTop = next.tableau[colIndex][next.tableau[colIndex].length - 1];
    if (!newTop.faceUp) newTop.faceUp = true;
  }
  next.moves = (next.moves || 0) + 1;
  return next;
}

export function moveWasteToTableau(state, toCol) {
  if (state.waste.length === 0) return state;
  const card = state.waste[state.waste.length - 1];
  const toColArr = state.tableau[toCol];
  const top = toColArr.length ? toColArr[toColArr.length - 1] : null;
  if (!canStackOnTableau(card, top)) return state;

  const next = JSON.parse(JSON.stringify(state));
  next.waste = next.waste.slice(0, -1);
  next.tableau[toCol] = [...next.tableau[toCol], card];
  next.moves = (next.moves || 0) + 1;
  return next;
}

export function moveWasteToFoundation(state) {
  if (state.waste.length === 0) return state;
  const card = state.waste[state.waste.length - 1];
  const fIndex = findFoundationDropIndex(card, state.foundations);
  if (fIndex < 0) return state;

  const next = JSON.parse(JSON.stringify(state));
  next.waste = next.waste.slice(0, -1);
  next.foundations[fIndex].push(card);
  next.moves = (next.moves || 0) + 1;
  return next;
}

export function moveFoundationToTableau(state, foundIndex, toCol) {
  const found = state.foundations[foundIndex];
  if (found.length === 0) return state;
  const card = found[found.length - 1];
  const toColArr = state.tableau[toCol];
  const top = toColArr.length ? toColArr[toColArr.length - 1] : null;
  if (!canStackOnTableau(card, top)) return state;

  const next = JSON.parse(JSON.stringify(state));
  next.foundations[foundIndex] = next.foundations[foundIndex].slice(0, -1);
  next.tableau[toCol] = [...next.tableau[toCol], card];
  next.moves = (next.moves || 0) + 1;
  return next;
}

export function isWon(state) {
  return state.foundations.every((f) => f.length === 13);
}

export function flipTableauCard(state, colIndex) {
  const col = state.tableau[colIndex];
  if (col.length === 0) return state;
  const top = col[col.length - 1];
  if (top.faceUp) return state;
  const next = JSON.parse(JSON.stringify(state));
  next.tableau[colIndex][next.tableau[colIndex].length - 1].faceUp = true;
  return next;
}
