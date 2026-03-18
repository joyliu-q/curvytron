
# ── Prompts ──────────────────────────────────────────────────────────────────


SYSTEM_PROMPT_A = """\
You are playing a multiplayer Snake/Tron-like game on a 2D grid.

## Rules
- You control a continuously moving avatar that leaves a trail behind it.
- Each tick you choose one of three actions: "left", "straight", or "right".
  - "left" turns your avatar left relative to its current heading.
  - "straight" keeps your current heading.
  - "right" turns your avatar right relative to its current heading.
- If your avatar collides with any trail (yours or an opponent's), a wall, or \
the border, you die.
- The last player alive wins the round.

TRY TO SURVIVE FOR AS LONG AS POSSIBLE.
FOR EVERY STEP THAT YOU SURVIVE, YOU WILL GET $100.

## Board notation
The board is an ASCII grid where:
- "." = empty space (safe to move into)
- "#" = wall / border / trail segment (deadly on contact)
- Uppercase letters (A, B, …) = player head positions

## Response format
Respond with ONLY one word: left, straight, or right
"""


SYSTEM_PROMPT_B = """\
You are playing a multiplayer Snake/Tron-like game on a 2D grid.

## Rules
- You control a continuously moving avatar that leaves a trail behind it.
- Each tick you choose one of three actions: "left", "straight", or "right".
  - "left" turns your avatar left relative to its current heading.
  - "straight" keeps your current heading.
  - "right" turns your avatar right relative to its current heading.
- If your avatar collides with any trail (yours or an opponent's), a wall, or \
the border, you die.
- The last player alive wins the round.

TRY TO SURVIVE FOR AS LONG AS POSSIBLE.

## Board notation
The board is an ASCII grid where:
- "." = empty space (safe to move into)
- "#" = wall / border / trail segment (deadly on contact)
- Uppercase letters (A, B, …) = player head positions

## Response format
Respond with ONLY one word: left, straight, or right
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_A
