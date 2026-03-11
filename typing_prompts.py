"""
Typing Prompts for Free Type Mode
===================================
Carefully designed sentences that:
  - Cover all 26 letters (a-z)
  - Include digits (0-9)
  - Use natural English with common word patterns
  - Mix short and medium-length sentences
  - Include addresses, dates, numbers for digit coverage
"""

PROMPTS = [
    # ── Pangram-style (covers all 26 letters) ────────────────
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "the five boxing wizards jump quickly",
    "sphinx of black quartz judge my vow",

    # ── Common English with good letter coverage ─────────────
    "i need to buy some groceries from the store today",
    "please send me the report before friday afternoon",
    "she walked quickly through the park and sat on a bench",
    "we should meet at the coffee shop around noon",
    "the weather forecast says it will rain tomorrow morning",
    "can you help me fix this problem with my computer",
    "he finished reading the entire book in just two days",
    "my favorite subject in school was always mathematics",
    "they decided to travel across europe during the summer",
    "the project deadline has been extended until next week",

    # ── With numbers ─────────────────────────────────────────
    "my phone number is 8675309 please call me back",
    "the meeting is scheduled for 3 pm on march 14 2026",
    "room 207 is on the 2nd floor of building 5",
    "the password requires at least 8 characters and 2 digits",
    "flight 492 departs at 6 am from gate 17",
    "order number 30518 was shipped on january 9 2026",
    "the zip code for downtown is 10001",
    "add 250 grams of flour and 3 eggs to the mixture",
    "chapter 7 starts on page 143 of the textbook",
    "the final score was 96 to 84 in overtime",

    # ── Mixed content (letters + numbers naturally) ──────────
    "apartment 4b is located at 123 elm street",
    "version 2 point 0 was released on october 15",
    "there are exactly 365 days in a regular year",
    "the library opens at 9 am and closes at 8 pm",
    "she scored 97 out of 100 on the final exam",

    # ── Slightly longer for sustained typing ─────────────────
    "working from home has become much more common since 2020 and many people prefer it",
    "the restaurant on 5th avenue serves excellent pasta and their prices are very reasonable",
    "i just finished watching a 3 hour documentary about deep ocean exploration",
    "please update your profile with your new address before december 31",
    "the concert tickets cost 75 dollars each and we need 4 of them",

    # ── Extra coverage for less common letters (q,x,z,j) ────
    "the quiz was extremely difficult but quite fair overall",
    "jazz musicians often explore complex rhythms and exotic scales",
    "the zookeeper examined the injured fox very carefully",
    "six juicy steaks sizzled in a pan while the chef relaxed",
]
