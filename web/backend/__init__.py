"""Web backend for the snake-dqn app (FastAPI). Replaces the PyQt5 GUI.

Server-authoritative: the Python engine (GameState + ApexPolicy) is the single
source of truth. The React frontend is a pure view that renders frames streamed
over a WebSocket. No game logic or model inference runs in the browser.
"""
