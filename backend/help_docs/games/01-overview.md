---
title: Games overview
order: 1
---

# Games overview

The **Games** page offers built-in games (e.g. **Solitaire** and **Lemonade Stand**) that you can play in the browser. Game state is saved per user in the browser (**localStorage**), so you can **Resume** later. Access to Games is controlled by capabilities: you need the **admin** role or the **feature.games.view** capability to see **Games** in the main navigation. This page describes how to open Games and what each game does.

---

## Accessing Games

- **Navigation** — If you have access, **Games** appears in the main navigation (top bar or sidebar). Click it to open the Games list. If you do not see Games, your account does not have the **feature.games.view** capability and you are not an admin; an admin can grant **feature.games.view** in **Settings > User Management** for your user.
- **After opening** — You see a list of available games as cards. Each card shows the game **title**, a short **description**, and a **Resume** chip if you have a saved game in progress. Click a card to start or resume that game.

---

## Solitaire (Klondike)

- **What it is** — Classic **Klondike** solitaire: build **foundations** (Ace through King by suit) and clear the **tableau** by moving cards in descending order, alternating colors.
- **How to play** — **Click a card** to select it, then **click a destination** (foundation or tableau pile) to move it. The game follows standard Klondike rules (e.g. only Kings go on empty tableau slots; you can draw from the stock).
- **Auto-complete** — When all cards are on the foundations, the game may **auto-complete** (e.g. a short animation or “You win”).
- **Resume** — Your progress is saved in the browser. If you leave and come back, the **Resume** chip appears on the Solitaire card; click to continue from the same state.

---

## Lemonade Stand

- **What it is** — A **30-day** business simulation: you run a lemonade stand by setting **price**, **buying ingredients**, and **buying upgrades**. **Weather** affects demand and sales.
- **How to play** — Each day you set how much to charge per cup, decide how much to spend on lemons and other ingredients, and optionally buy upgrades (e.g. better stand, signs). Then you see the day’s results (sales, profit, weather). After 30 days you see your total performance.
- **Resume** — Progress is saved in the browser. If you have a game in progress, the **Resume** chip appears on the Lemonade Stand card; click to continue from the last day you played.

---

## Game state and data

- **Where it’s stored** — Game state is saved in the browser’s **localStorage** under a key that includes your **user id** (e.g. `bastion_solitaire_<user_id>`). So each user has their own Solitaire and Lemonade Stand state on that device.
- **Resume chip** — A game shows the **Resume** chip when there is saved state that is not “game over” or “won” (e.g. Solitaire in progress, or Lemonade Stand before day 30). Clearing browser data for the site will remove saved games.
- **No server sync** — Games do not sync across devices or browsers; they are local to the browser and user.

---

## Summary

- **Games** is in the main navigation if you have **admin** or **feature.games.view**. Open it to see **Solitaire** (Klondike) and **Lemonade Stand**.
- **Solitaire**: click to select a card, click destination to move; **Resume** continues a saved game. **Lemonade Stand**: 30-day sim, set price and buy ingredients/upgrades; weather affects sales; **Resume** continues from last day.
- State is stored in **localStorage** per user; **Resume** chip appears when a game is in progress.

---

## Related

- **Settings overview** — Where admins manage users and capabilities.
- **Admin user management** — Granting **feature.games.view** for users.
