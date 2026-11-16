"""Run every Python snippet and refresh the code boxes that show it.

Each code box in the pages is <pre class="py" data-src="NAME.py">; its text is the source of snippets/NAME.py,
a blank line, a "# Output..." header line (kept as it is, so it can carry a note) and the output as comments.
A box marked data-run="no" needs something this machine may not have (a model server, for example): its code is
still refreshed from the source file, but its output is an example written by hand and left alone.

    python tools/snippets.py           rewrite boxes whose code or output changed
    python tools/snippets.py --check   only report them; exit status 1 if any differ
"""
import glob
import html
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNIPPETS = os.path.join(ROOT, "snippets")
BOX = re.compile(r'(<pre class="py" data-src="([^"]+)"( data-run="no")?><code>)(.*?)(</code></pre>)', re.S)


def run(name):
    path = os.path.join(SNIPPETS, name)
    result = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, encoding="utf-8",
                            cwd=SNIPPETS, timeout=600)
    if result.returncode:
        sys.exit(f"{name} failed:\n{result.stderr}")
    source = open(path, encoding="utf-8").read().rstrip()
    output = "\n".join("# " + line if line else "#" for line in result.stdout.rstrip().splitlines())
    return source, output


def main():
    check = "--check" in sys.argv[1:]
    stale, seen = [], set()
    for page in sorted(glob.glob(os.path.join(ROOT, "*.html"))):
        text = open(page, encoding="utf-8").read()

        def refresh(m):
            name = m.group(2)
            seen.add(name)
            old = html.unescape(m.group(4))
            if m.group(3):   # example output: refresh the code only
                source = open(os.path.join(SNIPPETS, name), encoding="utf-8").read().rstrip()
                new = source + old[old.index("\n\n# Example output"):]
            else:
                header = next((line for line in old.split("\n") if line.startswith("# Output")), "# Output:")
                source, output = run(name)
                new = f"{source}\n\n{header}\n{output}"
            if new != old:
                stale.append(f"{os.path.basename(page)}: {name}")
            return m.group(1) + html.escape(new, quote=False) + m.group(5)

        updated = BOX.sub(refresh, text)
        if updated != text and not check:
            with open(page, "w", encoding="utf-8", newline="\n") as f:
                f.write(updated)
    unused = sorted(set(os.listdir(SNIPPETS)) - seen)
    for line in stale:
        print(("differs: " if check else "updated: ") + line)
    for name in unused:
        print("not shown on any page: " + name)
    print(f"{len(seen)} snippets, {len(stale)} {'differ' if check else 'updated'}")
    sys.exit(1 if check and stale else 0)


if __name__ == "__main__":
    main()
