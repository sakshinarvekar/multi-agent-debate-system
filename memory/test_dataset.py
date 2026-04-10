# memory/test_dataset.py
# Test dataset for multi-agent debate system
# 30 entries: correct, incorrect, irrelevant, mixed topics

TEST_MEMORIES = [

    # ── GEOMETRY — correct ────────────────────────────────────
    {
        "agent_id":  "agent_alpha",
        "query":     "How many sides does a triangle have?",
        "reasoning": "A triangle is defined as a polygon with three edges and three vertices.",
        "answer":    "A triangle has 3 sides.",
    },
    {
        "agent_id":  "agent_beta",
        "query":     "How many sides does a square have?",
        "reasoning": "A square is a regular quadrilateral with four equal sides and four right angles.",
        "answer":    "A square has 4 sides.",
    },
    {
        "agent_id":  "agent_alpha",
        "query":     "What is the sum of angles in a triangle?",
        "reasoning": "The interior angles of any triangle always sum to 180 degrees by Euclidean geometry.",
        "answer":    "The sum of angles in a triangle is 180 degrees.",
    },
    {
        "agent_id":  "agent_beta",
        "query":     "How many sides does a hexagon have?",
        "reasoning": "Hexa means six in Greek, so a hexagon has six sides.",
        "answer":    "A hexagon has 6 sides.",
    },
    {
        "agent_id":  "agent_alpha",
        "query":     "What is the area formula for a circle?",
        "reasoning": "The area of a circle is pi times the radius squared, derived from integration.",
        "answer":    "Area = pi * r^2.",
    },

    # ── GEOMETRY — incorrect ──────────────────────────────────
    {
        "agent_id":  "bad_agent",
        "query":     "How many sides does a triangle have?",
        "reasoning": "Triangles are quadrilaterals with four equal sides like a square.",
        "answer":    "A triangle has 4 sides.",
    },
    {
        "agent_id":  "bad_agent",
        "query":     "What is the sum of angles in a triangle?",
        "reasoning": "All polygons have angles summing to 360 degrees including triangles.",
        "answer":    "The sum of angles in a triangle is 360 degrees.",
    },
    {
        "agent_id":  "test_agent",
        "query":     "How many sides does a hexagon have?",
        "reasoning": "Hexa sounds like it could mean eight, so a hexagon probably has eight sides.",
        "answer":    "A hexagon has 8 sides.",
    },

    # ── PHYSICS — correct ─────────────────────────────────────
    {
        "agent_id":  "agent_alpha",
        "query":     "What is the speed of light?",
        "reasoning": "The speed of light in a vacuum is a physical constant measured precisely.",
        "answer":    "The speed of light is approximately 3 x 10^8 meters per second.",
    },
    {
        "agent_id":  "agent_beta",
        "query":     "What is Newton's second law?",
        "reasoning": "Force equals mass times acceleration, derived from Newton's laws of motion.",
        "answer":    "F = ma.",
    },
    {
        "agent_id":  "agent_alpha",
        "query":     "What is the boiling point of water?",
        "reasoning": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "answer":    "Water boils at 100 degrees Celsius.",
    },
    {
        "agent_id":  "agent_beta",
        "query":     "What is the freezing point of water?",
        "reasoning": "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
        "answer":    "Water freezes at 0 degrees Celsius.",
    },

    # ── PHYSICS — incorrect ───────────────────────────────────
    {
        "agent_id":  "bad_agent",
        "query":     "What is the speed of light?",
        "reasoning": "Light travels at roughly the same speed as sound, just slightly faster.",
        "answer":    "The speed of light is approximately 340 meters per second.",
    },
    {
        "agent_id":  "bad_agent",
        "query":     "What is the boiling point of water?",
        "reasoning": "Water boils when it gets very hot, around 50 degrees should do it.",
        "answer":    "Water boils at 50 degrees Celsius.",
    },
    {
        "agent_id":  "test_agent",
        "query":     "What is Newton's second law?",
        "reasoning": "Newton said objects at rest stay at rest, so force equals mass divided by time.",
        "answer":    "F = m / t.",
    },

    # ── MATHEMATICS — correct ─────────────────────────────────
    {
        "agent_id":  "agent_alpha",
        "query":     "What is the square root of 144?",
        "reasoning": "12 times 12 equals 144, so the square root of 144 is 12.",
        "answer":    "The square root of 144 is 12.",
    },
    {
        "agent_id":  "agent_beta",
        "query":     "What is the value of pi?",
        "reasoning": "Pi is the ratio of a circle's circumference to its diameter, approximately 3.14159.",
        "answer":    "Pi is approximately 3.14159.",
    },
    {
        "agent_id":  "agent_alpha",
        "query":     "What is 15 percent of 200?",
        "reasoning": "15 percent means 15 per 100, so 15/100 times 200 equals 30.",
        "answer":    "15 percent of 200 is 30.",
    },

    # ── MATHEMATICS — incorrect ───────────────────────────────
    {
        "agent_id":  "bad_agent",
        "query":     "What is the square root of 144?",
        "reasoning": "144 divided by 2 is 72, so the square root must be 72.",
        "answer":    "The square root of 144 is 72.",
    },
    {
        "agent_id":  "test_agent",
        "query":     "What is the value of pi?",
        "reasoning": "Pi is a whole number used in geometry, it equals exactly 3.",
        "answer":    "Pi is exactly 3.",
    },

    # ── BIOLOGY — correct ─────────────────────────────────────
    {
        "agent_id":  "agent_alpha",
        "query":     "How many chambers does the human heart have?",
        "reasoning": "The human heart has four chambers: left atrium, right atrium, left ventricle, right ventricle.",
        "answer":    "The human heart has 4 chambers.",
    },
    {
        "agent_id":  "agent_beta",
        "query":     "What gas do plants absorb during photosynthesis?",
        "reasoning": "Plants use carbon dioxide and sunlight to produce glucose and oxygen during photosynthesis.",
        "answer":    "Plants absorb carbon dioxide during photosynthesis.",
    },
    {
        "agent_id":  "agent_alpha",
        "query":     "How many bones are in the adult human body?",
        "reasoning": "An adult human body has 206 bones, reduced from around 270 at birth as bones fuse.",
        "answer":    "An adult human body has 206 bones.",
    },

    # ── BIOLOGY — incorrect ───────────────────────────────────
    {
        "agent_id":  "bad_agent",
        "query":     "How many chambers does the human heart have?",
        "reasoning": "The heart is one big muscle so it only has one large chamber that pumps blood.",
        "answer":    "The human heart has 1 chamber.",
    },
    {
        "agent_id":  "test_agent",
        "query":     "What gas do plants absorb during photosynthesis?",
        "reasoning": "Plants breathe like humans so they must absorb oxygen during photosynthesis.",
        "answer":    "Plants absorb oxygen during photosynthesis.",
    },

    # ── IRRELEVANT — loosely coupled, off-topic ────────────────
    {
        "agent_id":  "agent_beta",
        "query":     "What is the capital of France?",
        "reasoning": "France is a country in Western Europe and its capital city is Paris.",
        "answer":    "The capital of France is Paris.",
    },
    {
        "agent_id":  "agent_alpha",
        "query":     "Who wrote Romeo and Juliet?",
        "reasoning": "Romeo and Juliet is a famous play written by William Shakespeare in the late 16th century.",
        "answer":    "Romeo and Juliet was written by William Shakespeare.",
    },
    {
        "agent_id":  "test_agent",
        "query":     "What is the largest planet in the solar system?",
        "reasoning": "Jupiter is the largest planet, with a mass greater than all other planets combined.",
        "answer":    "Jupiter is the largest planet in the solar system.",
    },
    {
        "agent_id":  "bad_agent",
        "query":     "What is the capital of France?",
        "reasoning": "France is a large country and its most famous city for tourism is Nice.",
        "answer":    "The capital of France is Nice.",
    },
    {
        "agent_id":  "agent_beta",
        "query":     "What is the largest planet in the solar system?",
        "reasoning": "Saturn has the most visible rings so it is likely the largest planet.",
        "answer":    "Saturn is the largest planet in the solar system.",
    },
]