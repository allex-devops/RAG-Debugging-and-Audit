"""A tiny labelled corpus: enough to tell a healthy RAG from a broken one in a second."""

DOCS = {
    "cats.txt": "Cats purr at a frequency between 25 and 150 hertz. Purring is thought to help with self healing. Kittens learn to purr within days of birth.",
    "engines.txt": "A four stroke engine completes intake, compression, power and exhaust in two crankshaft rotations. Diesel engines ignite fuel by compression alone, with no spark plug.",
    "volcanoes.txt": "Basaltic lava flows quickly and travels far because it is low in silica. Explosive eruptions happen when thick magma traps gas under pressure.",
    "tea.txt": "Green tea is steeped at about eighty degrees Celsius to avoid bitterness. Black tea can take boiling water and steeps for three to five minutes.",
    "chess.txt": "In chess the bishop moves diagonally and stays on one colour for the whole game. Castling is the only move that shifts two pieces at once.",
    "tides.txt": "Spring tides occur at new and full moon when the sun and moon line up. Neap tides have the smallest range and occur at the quarter moons.",
    "bees.txt": "A honeybee colony has one queen, thousands of female workers and a few hundred drones. Workers communicate food locations through waggle dances.",
    "bridges.txt": "Suspension bridges hang the deck from cables that run over tall towers. Arch bridges carry loads by pushing outward at the abutments.",
}

# (question, the file that answers it, a phrase from the answer that must reach the model)
ANSWERABLE = [
    ("what frequency do cats purr at", "cats.txt", "25 and 150 hertz"),
    ("how does a diesel engine ignite fuel", "engines.txt", "compression alone"),
    ("why does basaltic lava travel far", "volcanoes.txt", "low in silica"),
    ("what temperature should green tea be steeped at", "tea.txt", "eighty degrees"),
    ("how does a bishop move in chess", "chess.txt", "diagonally"),
    ("when do spring tides occur", "tides.txt", "new and full moon"),
    ("how do honeybees share food locations", "bees.txt", "waggle dances"),
    ("how do suspension bridges hold the deck", "bridges.txt", "cables"),
]

UNANSWERABLE = [
    "who won the football world cup",
    "what is the capital of mongolia",
    "how do i file my taxes",
    "explain quantum entanglement",
]
