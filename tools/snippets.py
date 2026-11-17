"""Run every Python snippet and refresh the code boxes that show it.

Each code box in the pages is <pre class="py" data-src="NAME.py">; its text is the source of snippets/NAME.py,
a blank line, a "# Output..." header line (kept as it is, so it can carry a note) and the output as comments.
A box marked data-run="no" needs something this machine may not have (a model server, for example): its code is
still refreshed from the source file, but its output is an example written by hand and left alone.

Snippets target Python 3.11, and the script refuses to run under another version. Before running anything it checks
every file in snippets/: it compiles, each block is indented 4 spaces deeper than the one around it, and there are
no tabs or trailing spaces.

    python tools/snippets.py           rewrite boxes whose code or output changed
    python tools/snippets.py --check   only report them; exit status 1 if any differ
"""
import glob
import html
import io
import os
import re
import subprocess
import sys
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNIPPETS = os.path.join(ROOT, "snippets")
BOX = re.compile(r'(<pre class="py" data-src="([^"]+)"( data-run="no")?><code>)(.*?)(</code></pre>)', re.S)
PYTHON = (3, 11)


def lint(path):
    """Problems with one snippet file: syntax errors, indentation, tabs, trailing spaces. Line endings are left to git."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    problems = []
    try:
        compile(src, path, "exec")
    except SyntaxError as e:
        return problems + [f"line {e.lineno}: {e.msg}"]
    for i, line in enumerate(src.split("\n"), 1):
        if "\t" in line:
            problems.append(f"line {i}: tab")
        if line != line.rstrip():
            problems.append(f"line {i}: trailing space")
    depth = [0]
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.INDENT:
            if len(tok.string) - depth[-1] != 4:
                problems.append(f"line {tok.start[0]}: indented by {len(tok.string) - depth[-1]}, not 4 spaces")
            depth.append(len(tok.string))
        elif tok.type == tokenize.DEDENT:
            depth.pop()
    return problems


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
    if sys.version_info[:2] != PYTHON:
        sys.exit(f"Run this with Python {PYTHON[0]}.{PYTHON[1]}; this is {sys.version.split()[0]}.")
    problems = [f"{name}: {p}" for name in sorted(os.listdir(SNIPPETS)) if name.endswith(".py")
                for p in lint(os.path.join(SNIPPETS, name))]
    if problems:
        sys.exit("\n".join(problems))
    stale, seen = [], set()
    for page in sorted(glob.glob(os.path.join(ROOT, "site", "*.html"))):
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
