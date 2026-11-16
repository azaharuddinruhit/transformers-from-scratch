import random
from collections import Counter, defaultdict

NAMES = """aaron abdul adam adrian aisha akira alba alex alice alina amara amir amy ana andrea andrei anika anna anton
arjun ava ayesha beatriz ben bianca boris bruno carla carlos carmen caroline chen chloe clara daniel david diana diego
dmitri elena eli elias elif elise ella emil emily emma erik esther ethan eva farah felix fiona frida gabriel grace hana
hannah hassan hector helen hugo ibrahim ida imran ines ingrid irene isaac isabel ivan jack jacob james jana jasmine
javier jonas jose julia julian kai karim karin karl kate kenji kira lara laura layla leo leon lena liam lina linus lucas
lucia luis luna maja malik marco maria marina mark marta martin maya mia mila mina miriam mohammed nadia naomi nathan
nina noah nora olga oliver omar oscar pablo paula pedro peter priya rafael rahul rania ravi rina rosa ruby rupert sara
sami samuel sana sofia sonia stefan sven tara theo thomas tomas valentina vera victor viktor william yara yasmin yusuf
zara zoe""".split()
END = "."  # marks the start and the end of a name


def train(k):
    """For every run of k letters, count which letter came next."""
    table = defaultdict(Counter)
    for name in NAMES:
        s = END * k + name + END
        for i in range(k, len(s)):
            table[s[i - k:i]][s[i]] += 1
    return table


def generate(table, k, rng):
    """Draw one letter at a time from the counts until the end marker is drawn."""
    name = ""
    while True:
        counts = table[(END * k + name)[-k:]]
        letter = rng.choices(list(counts), weights=list(counts.values()))[0]
        if letter == END:
            return name
        name += letter


table = train(2)
after = table["ma"]
total = sum(after.values())
print("after 'ma':", {c: f"{n}/{total}" for c, n in after.most_common()})

rng = random.Random(0)
known = set(NAMES)
for k in (1, 2, 3, 4):
    table = train(k)
    names = [generate(table, k, rng) for _ in range(1000)]
    copies = sum(n in known for n in names) / len(names)
    print(f"context {k}: {copies:5.1%} copies   e.g. {' '.join(names[:6])}")
